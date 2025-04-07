import os
import io
import time
import base64
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import torch
from openai import OpenAI

class VideoGameStateProcessor:
    """
    Processes video frames and game state data to enable reasoning models
    to determine the next action in arbitrary 3D games.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize video processor with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__ + ".VideoGameStateProcessor")
        
        # Configuration parameters
        self.buffer_size = config.get("video_buffer_size", 5)
        self.frame_interval = config.get("frame_interval", 2)
        self.frame_counter = 0
        self.change_threshold = config.get("change_threshold", 0.15)
        
        # Initialize OpenAI client for vision analysis
        openai_api_key = config.get("openai_api_key", os.environ.get("OPENAI_API_KEY"))
        if not openai_api_key:
            self.logger.warning("No OpenAI API key provided, VLM analysis will not be available")
        self.vlm_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # Buffers for frames and state data
        self.frame_buffer = []  # List of {"frame": PIL.Image, "timestamp": float}
        self.state_buffer = []  # List of {"state": Dict, "timestamp": float}
        
        # Last significant change timestamp
        self.last_significant_change = time.time()
        
        self.logger.info("Video Game State Processor initialized")
    
    def add_frame(self, frame: Image.Image, game_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a frame to the buffer, optionally with corresponding game state data
        
        Args:
            frame: PIL Image of the current game frame
            game_state: Dictionary containing game state information from Unity
        """
        self.frame_counter += 1
        
        # Only process every nth frame to save resources
        if self.frame_counter % self.frame_interval == 0:
            timestamp = time.time()
            
            # Add frame to buffer
            self.frame_buffer.append({"frame": frame, "timestamp": timestamp})
            
            # Add game state if provided
            if game_state:
                self.state_buffer.append({"state": game_state, "timestamp": timestamp})
            
            # Keep buffers at desired size
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)
            if game_state and len(self.state_buffer) > self.buffer_size:
                self.state_buffer.pop(0)
            
            # Check for significant changes
            if len(self.frame_buffer) >= 2:
                if self._detect_significant_change(self.frame_buffer[-2]["frame"], self.frame_buffer[-1]["frame"]):
                    self.last_significant_change = timestamp
    
    def _detect_significant_change(self, prev_frame: Image.Image, curr_frame: Image.Image) -> bool:
        """
        Detect if there is a significant change between two frames
        
        Args:
            prev_frame: Previous frame as PIL Image
            curr_frame: Current frame as PIL Image
            
        Returns:
            True if significant change detected, False otherwise
        """
        # Convert to numpy arrays for comparison
        # Resize to smaller dimension for faster processing
        size = (224, 224)
        prev_np = np.array(prev_frame.resize(size))
        curr_np = np.array(curr_frame.resize(size))
        
        # Handle different image formats
        if len(prev_np.shape) == 2:  # Grayscale
            prev_np = np.stack([prev_np, prev_np, prev_np], axis=2)
        if len(curr_np.shape) == 2:  # Grayscale
            curr_np = np.stack([curr_np, curr_np, curr_np], axis=2)
            
        # Ensure we're comparing RGB only
        if prev_np.shape[2] > 3:
            prev_np = prev_np[:, :, :3]
        if curr_np.shape[2] > 3:
            curr_np = curr_np[:, :, :3]
        
        # Calculate mean squared error
        try:
            mse = np.mean(np.square(prev_np.astype(float) - curr_np.astype(float)))
            normalized_mse = mse / (255.0 * 255.0)  # Normalize to 0-1 range
            return normalized_mse > self.change_threshold
        except Exception as e:
            self.logger.error(f"Error detecting frame change: {str(e)}")
            return False
    
    def analyze_game_situation(self, task_context: str) -> Dict[str, Any]:
        """
        Analyze current game situation using both visual and state data
        
        Args:
            task_context: Description of the current test task
            
        Returns:
            Dictionary with analysis results
        """
        if not self.frame_buffer or not self.vlm_client:
            return {"error": "No frames available or VLM client not initialized"}
        
        try:
            # Prepare visual data
            visual_content = self._prepare_frame_content()
            
            # Prepare state data
            state_summary = self._summarize_state_data() if self.state_buffer else "No state data available"
            
            # Combine for analysis
            prompt = f"""
            Task Context: {task_context}
            
            Game State Summary:
            {state_summary}
            
            Analyze the provided 3D game frames and state data to:
            1. Describe the current game situation in detail
            2. Identify any potential bugs or unexpected behaviors
            3. Explain the relationship between visible elements and game state
            
            Focus on 3D spatial relationships, character animations, physics interactions,
            and any visual artifacts that might indicate bugs.
            """
            
            content = [{"type": "text", "text": prompt}] + visual_content
            
            # Call VLM API
            response = self.vlm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=800
            )
            
            return {
                "analysis": response.choices[0].message.content,
                "frames_analyzed": len(self.frame_buffer),
                "state_data_analyzed": len(self.state_buffer)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing game situation: {str(e)}")
            return {"error": str(e)}
    
    def determine_next_action(self, game_description: str, test_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the next action to take based on video analysis
        
        Args:
            game_description: General description of the game
            test_task: Dictionary with test task information
            
        Returns:
            Dictionary with the recommended action
        """
        if not self.frame_buffer or not self.vlm_client:
            return {"action": "wait", "reason": "Insufficient frames for analysis or no VLM client"}
        
        try:
            # Check if we need an exploratory action due to lack of changes
            time_since_change = time.time() - self.last_significant_change
            if time_since_change > 5.0:  # 5 seconds without significant changes
                return self._plan_exploration_action(game_description, test_task)
            
            # Prepare frames for API
            visual_content = self._prepare_frame_content()
            
            # Create prompt with game context and test objectives
            text_content = f"""
            Game Description: {game_description}
            
            Test Task: {test_task.get('description', 'No description available')}
            Expected Outcome: {test_task.get('expected_outcome', 'No expected outcome specified')}
            
            Based on the sequence of frames showing the current 3D game state, determine the optimal next action to:
            1. Make progress toward the test objective
            2. Explore potentially problematic interactions
            3. Test edge cases that might reveal bugs
            
            Focus on 3D navigation, physics interactions, and object manipulation relevant to this 3D game.
            
            Return your response as a structured action plan with specific steps the agent should take.
            """
            
            # Combine text and images
            content = [{"type": "text", "text": text_content}] + visual_content
            
            # Call VLM API
            response = self.vlm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=500
            )
            
            # Parse the response to extract the action
            action_text = response.choices[0].message.content
            return self._parse_action(action_text)
            
        except Exception as e:
            self.logger.error(f"Error determining next action: {str(e)}")
            return {"action": "error", "reason": str(e)}
    
    def _plan_exploration_action(self, game_description: str, test_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Plan an exploratory action when game state has been stable too long
        
        Args:
            game_description: Description of the game
            test_task: Dictionary with test task details
            
        Returns:
            Dictionary with the exploratory action details
        """
        if not self.frame_buffer or not self.vlm_client:
            return {"action": "random_movement", "reason": "No frames or VLM client available"}
        
        try:
            # Get most recent frame
            current_frame = self.frame_buffer[-1]["frame"]
            current_b64 = self._frame_to_base64(current_frame)
            
            # Construct prompt for exploration
            prompt = f"""
            Game Description: {game_description}
            
            Test Task: {test_task.get('description', 'No description available')}
            
            The game has been in a stable state with no significant changes for over 5 seconds.
            Based on the current frame shown, suggest an exploratory action that might:
            
            1. Progress the game state
            2. Test for potential bugs or edge cases
            3. Interact with visible 3D game elements
            
            Focus on 3D spatial navigation, physics interactions, and input combinations that might 
            reveal unexpected behaviors. Be specific about movements, camera angles, and interaction targets.
            """
            
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{current_b64}"}}
            ]
            
            # Call VLM API
            response = self.vlm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=300
            )
            
            action_text = response.choices[0].message.content
            return {"action": "exploration", "description": action_text}
            
        except Exception as e:
            self.logger.error(f"Error planning exploration action: {str(e)}")
            return {"action": "random_movement", "reason": str(e)}
    
    def _prepare_frame_content(self) -> List[Dict[str, Any]]:
        """
        Prepare frame content for API submission
        
        Returns:
            List of frame content dictionaries for the VLM API
        """
        # Select a subset of frames if buffer is full
        if len(self.frame_buffer) >= self.buffer_size:
            # Use first, middle, and last frames for temporal context
            indices = [0, self.buffer_size // 2, self.buffer_size - 1]
            frames_to_use = [self.frame_buffer[i] for i in indices]
        else:
            # Use all available frames
            frames_to_use = self.frame_buffer
        
        frame_contents = []
        for idx, frame_data in enumerate(frames_to_use):
            frame = frame_data["frame"]
            frame_b64 = self._frame_to_base64(frame)
            description = f"Frame {idx+1}/{len(frames_to_use)}"
            
            frame_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
            })
        
        return frame_contents
    
    def _summarize_state_data(self) -> str:
        """
        Create a text summary of the game state data
        
        Returns:
            Text summary of the game state data
        """
        if not self.state_buffer:
            return "No state data available"
        
        # Extract key information from state buffer
        summary_parts = []
        
        for idx, state_data in enumerate(self.state_buffer):
            state = state_data["state"]
            timestamp = state_data["timestamp"]
            formatted_time = time.strftime("%H:%M:%S", time.localtime(timestamp))
            
            summary = f"Frame {idx} ({formatted_time}):\n"
            
            # Add relevant state data based on what's available
            if isinstance(state, dict):
                # Handle player data
                if "player" in state:
                    player = state["player"]
                    if isinstance(player, dict):
                        summary += "- Player:\n"
                        for k, v in player.items():
                            summary += f"  * {k}: {v}\n"
                    else:
                        summary += f"- Player: {player}\n"
                
                # Handle object data
                if "objects" in state:
                    objects = state["objects"]
                    if isinstance(objects, list):
                        summary += f"- Objects: {len(objects)} items\n"
                        for i, obj in enumerate(objects[:3]):  # Limit to first 3 objects
                            if isinstance(obj, dict):
                                obj_type = obj.get("type", "unknown")
                                obj_pos = obj.get("position", "unknown")
                                summary += f"  * {obj_type} at {obj_pos}\n"
                            else:
                                summary += f"  * Object {i}: {obj}\n"
                    else:
                        summary += f"- Objects: {objects}\n"
                
                # Handle physics/collision data
                if "physics" in state:
                    physics = state["physics"]
                    summary += "- Physics:\n"
                    if isinstance(physics, dict):
                        for k, v in physics.items():
                            summary += f"  * {k}: {v}\n"
                    else:
                        summary += f"  * {physics}\n"
                
                # Handle any other top-level keys
                for key, value in state.items():
                    if key not in ["player", "objects", "physics"]:
                        summary += f"- {key}: {value}\n"
            else:
                # If state is not a dictionary, just add it as is
                summary += f"- State: {state}\n"
                
            summary_parts.append(summary)
        
        return "\n".join(summary_parts)
    
    def _parse_action(self, action_text: str) -> Dict[str, Any]:
        """
        Parse the model output into a structured action
        
        Args:
            action_text: Text response from the VLM
            
        Returns:
            Dictionary with structured action information
        """
        # Simple parsing logic - can be enhanced with more structure
        action_lower = action_text.lower()
        
        # Navigation actions
        if "move" in action_lower or "navigate" in action_lower or "walk" in action_lower:
            direction = self._extract_direction(action_text)
            return {"action": "move", "direction": direction, "description": action_text}
        
        # Interaction actions
        elif "interact" in action_lower or "click" in action_lower or "select" in action_lower:
            target = self._extract_target(action_text)
            return {"action": "interact", "target": target, "description": action_text}
        
        # Camera actions
        elif "camera" in action_lower or "look" in action_lower or "view" in action_lower:
            return {"action": "camera", "description": action_text}
        
        # Wait/observe actions
        elif "wait" in action_lower or "observe" in action_lower:
            duration = 2  # Default 2 seconds
            # Try to extract a duration
            import re
            duration_match = re.search(r'(\d+)(?:\s*)(second|sec)', action_lower)
            if duration_match:
                try:
                    duration = int(duration_match.group(1))
                except ValueError:
                    pass
            return {"action": "wait", "duration": duration, "description": action_text}
        
        # Default to a custom action
        else:
            return {"action": "custom", "description": action_text}
    
    def _extract_direction(self, text: str) -> str:
        """Extract movement direction from text"""
        text_lower = text.lower()
        directions = ["forward", "backward", "left", "right", "up", "down", "north", "south", "east", "west"]
        
        for direction in directions:
            if direction in text_lower:
                return direction
                
        return "forward"  # Default direction
    
    def _extract_target(self, text: str) -> str:
        """Extract interaction target from text"""
        # This could be enhanced with NLP for better extraction
        import re
        
        # Look for patterns like "click/interact with [target]"
        patterns = [
            r'(?:click|interact with|select|use|activate)\s+(?:the\s+)?([a-zA-Z0-9_\- ]+)',
            r'(?:clicking|interacting with|selecting|using)\s+(?:the\s+)?([a-zA-Z0-9_\- ]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1).strip()
        
        return "nearest object"  # Default target
    
    def _frame_to_base64(self, frame: Image.Image) -> str:
        """
        Convert PIL Image to base64 string
        
        Args:
            frame: PIL Image to convert
            
        Returns:
            Base64 encoded string
        """
        # Resize image if it's large to reduce API payload size
        max_size = (640, 480)
        if frame.width > max_size[0] or frame.height > max_size[1]:
            frame = frame.resize(max_size, Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
            
        # Convert RGBA to RGB mode if needed
        if frame.mode == 'RGBA':
            frame = frame.convert('RGB')
            
        buffer = io.BytesIO()
        # Use JPEG instead of PNG for smaller size, with good quality
        frame.save(buffer, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Enhanced observer that captures frames for video processing
class VideoObserver:
    """
    Observer that captures frames from a game for video analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize video observer with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__ + ".VideoObserver")
        
        # Get configuration
        self.capture_fps = config.get("capture_fps", 10)
        self.resize_shape = config.get("resize_shape", (640, 480))
        
        # Initialize processor
        self.processor = VideoGameStateProcessor(config)
        
        # Schedule settings
        self.last_capture_time = 0
        self.capture_interval = 1.0 / self.capture_fps
        
        self.logger.info(f"VideoObserver initialized with target {self.capture_fps} FPS")
    
    def capture_frame(self) -> Optional[Image.Image]:
        """
        Capture a frame from the game (to be implemented by subclasses)
        
        Returns:
            PIL Image of the captured frame or None if capture failed
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def update(self, game_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the observer by capturing a new frame if it's time
        
        Args:
            game_state: Optional game state data to associate with the frame
        """
        current_time = time.time()
        
        # Check if it's time to capture a new frame
        if current_time - self.last_capture_time >= self.capture_interval:
            # Capture frame
            frame = self.capture_frame()
            
            # Process frame if capture successful
            if frame is not None:
                self.processor.add_frame(frame, game_state)
                self.last_capture_time = current_time
    
    def get_processor(self) -> VideoGameStateProcessor:
        """
        Get the video processor
        
        Returns:
            Video processor instance
        """
        return self.processor

# Unity-specific implementation of the video observer
class UnityVideoObserver(VideoObserver):
    """
    Observer that captures frames from a Unity game
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Unity video observer"""
        super().__init__(config)
        
        # Unity-specific configuration
        self.unity_window_title = config.get("unity_window_title", "Unity Game")
        self.fallback_to_fullscreen = config.get("fallback_to_fullscreen", True)
        
        # Import screenshot capability
        try:
            import pyautogui
            self.screenshot_module = pyautogui
            self.logger.info("Using pyautogui for screenshots")
        except ImportError:
            try:
                import pyscreenshot as ImageGrab
                self.screenshot_module = ImageGrab
                self.logger.info("Using pyscreenshot for screenshots")
            except ImportError:
                self.logger.error("No screenshot module available. Install either pyautogui or pyscreenshot.")
                self.screenshot_module = None
    
    def capture_frame(self) -> Optional[Image.Image]:
        """
        Capture a frame from the Unity game window
        
        Returns:
            PIL Image of the captured frame or None if capture failed
        """
        if not self.screenshot_module:
            self.logger.error("No screenshot module available")
            return None
        
        try:
            # Attempt to capture the Unity window specifically (platform-dependent)
            try:
                if hasattr(self.screenshot_module, "screenshot"):
                    # PyAutoGUI approach
                    screenshot = self.screenshot_module.screenshot()
                else:
                    # Pyscreenshot approach
                    screenshot = self.screenshot_module.grab()
                
                # Resize screenshot
                if self.resize_shape:
                    screenshot = screenshot.resize(self.resize_shape)
                
                return screenshot
                
            except Exception as e:
                self.logger.warning(f"Error capturing specific window: {str(e)}")
                
                # Fall back to full screen if enabled
                if self.fallback_to_fullscreen:
                    self.logger.info("Falling back to full screen capture")
                    if hasattr(self.screenshot_module, "screenshot"):
                        screenshot = self.screenshot_module.screenshot()
                    else:
                        screenshot = self.screenshot_module.grab()
                    
                    # Resize screenshot
                    if self.resize_shape:
                        screenshot = screenshot.resize(self.resize_shape)
                    
                    return screenshot
                else:
                    return None
                
        except Exception as e:
            self.logger.error(f"Error capturing frame: {str(e)}")
            return None
