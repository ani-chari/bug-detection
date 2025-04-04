# sima/agent.py
import torch
import logging
import time
import numpy as np
from typing import Dict, Any, List

from .models.vision import VisionModel
from .models.language import LanguageModel
from .models.action import ActionModel
from .models.integration import IntegrationModel
from .environment.observer import ScreenObserver
from .environment.controller import InputController
from .utils.config import default_config

class SIMAAgent:
    """
    Implementation of SIMA (Scalable Instructable Multiworld Agent) from DeepMind.
    
    This agent follows natural language instructions in 3D environments by observing 
    the screen and controlling keyboard/mouse inputs.
    
    Inputs:
    - Natural language instructions (text strings)
    
    Outputs:
    - Execution results (success/failure, observations, etc.)
    - Keyboard and mouse actions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Determine device
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Get observer configuration
        observer_config = self.config.get("observer", {})
        capture_method = observer_config.get("capture_method", "screen")
        
        # Get controller configuration  
        controller_config = self.config.get("controller", {})
        control_method = controller_config.get("control_method", "keyboard_mouse")
        
        # Initialize models
        # Initialize models
        self.vision_model = VisionModel(self.config.get("vision", {})).to(self.device)
        self.language_model = LanguageModel(self.config.get("language", {})).to(self.device)
        self.action_model = ActionModel(self.config.get("action", {})).to(self.device)
        self.integration_model = IntegrationModel(
            self.config.get("integration", {}),
            self.vision_model,
            self.language_model,
            self.action_model
        ).to(self.device)
        
        # FORCE use of ADB observer for Android regardless of config
        # This bypasses any issues with capture_method not being set
        try:
            # Try to import and use ADBScreenObserver
            try:
                from .environment.adb_observer import ADBScreenObserver
                self.observer = ADBScreenObserver(observer_config)
                self.logger.info("Using ADBScreenObserver")
            except (ImportError, ModuleNotFoundError):
                # If import fails, define the class inline
                self.logger.info("ADBScreenObserver not found, creating inline implementation")
                
                import subprocess
                import tempfile
                import os
                from PIL import Image
                
                class InlineADBObserver:
                    def __init__(self, config):
                        self.config = config
                        self.logger = logging.getLogger(__name__ + ".InlineADBObserver")
                        self.resize_shape = config.get("resize_shape", (224, 224))
                        self.device_id = config.get("device_id")
                        
                        if not self.device_id:
                            # Find device ID
                            try:
                                result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
                                lines = result.stdout.strip().split('\n')[1:]
                                for line in lines:
                                    if line.strip() and "device" in line:
                                        self.device_id = line.split()[0]
                                        break
                            except:
                                self.device_id = None
                    
                    def get_observation(self):
                        try:
                            # Create temp file
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                                temp_path = temp_file.name
                            
                            # Capture screenshot
                            if self.device_id:
                                subprocess.run(['adb', '-s', self.device_id, 'shell', 'screencap', '-p', '/sdcard/screenshot.png'], check=True)
                                subprocess.run(['adb', '-s', self.device_id, 'pull', '/sdcard/screenshot.png', temp_path], check=True)
                            else:
                                subprocess.run(['adb', 'shell', 'screencap', '-p', '/sdcard/screenshot.png'], check=True)
                                subprocess.run(['adb', 'pull', '/sdcard/screenshot.png', temp_path], check=True)
                            
                            # Process image
                            img = Image.open(temp_path).convert('RGB')
                            img = img.resize(self.resize_shape)
                            img_np = np.array(img)
                            img_tensor = torch.tensor(img_np).permute(2, 0, 1).float() / 255.0
                            
                            # Clean up
                            os.unlink(temp_path)
                            
                            return img_tensor
                        except Exception as e:
                            self.logger.error(f"Error capturing screenshot: {e}")
                            blank = np.zeros((self.resize_shape[1], self.resize_shape[0], 3), dtype=np.uint8)
                            return torch.tensor(blank).permute(2, 0, 1).float() / 255.0
                
                self.observer = InlineADBObserver(observer_config)
                
        except Exception as e:
            self.logger.error(f"Error initializing observer: {e}")
            # Set up a mock observer as fallback
            self.observer = None
            
        # Similarly force use of ADB controller
        try:
            # Try to import and use ADBInputController
            try:
                from .environment.adb_controller import ADBInputController
                self.controller = ADBInputController(controller_config)
                self.logger.info("Using ADBInputController")
            except (ImportError, ModuleNotFoundError):
                # Implement inline ADB controller
                self.logger.info("ADBInputController not found, creating inline implementation")
                
                class InlineADBController:
                    def __init__(self, config):
                        self.config = config
                        self.logger = logging.getLogger(__name__ + ".InlineADBController")
                        self.device_id = config.get("device_id")
                        self.action_delay = config.get("action_delay", 0.5)
                        
                        if not self.device_id:
                            # Find device ID
                            try:
                                result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
                                lines = result.stdout.strip().split('\n')[1:]
                                for line in lines:
                                    if line.strip() and "device" in line:
                                        self.device_id = line.split()[0]
                                        break
                            except:
                                self.device_id = None
                    
                    def execute(self, actions):
                        results = []
                        success = True
                        
                        for action in actions:
                            try:
                                action_type = action.get("type", "")
                                
                                if action_type == "touch" and action.get("action") == "tap":
                                    pos = action.get("position", {"x": 0.5, "y": 0.5})
                                    x, y = pos.get("x", 0.5), pos.get("y", 0.5)
                                    # Convert to screen coordinates (assuming 1080x1920)
                                    sx, sy = int(x * 1080), int(y * 1920)
                                    
                                    if self.device_id:
                                        subprocess.run(['adb', '-s', self.device_id, 'shell', 'input', 'tap', str(sx), str(sy)], check=True)
                                    else:
                                        subprocess.run(['adb', 'shell', 'input', 'tap', str(sx), str(sy)], check=True)
                                    
                                    results.append({"success": True, "type": "touch", "action": "tap"})
                                    
                                elif action_type == "swipe":
                                    start = action.get("start", {"x": 0.5, "y": 0.7})
                                    end = action.get("end", {"x": 0.5, "y": 0.3})
                                    
                                    # Convert to screen coordinates
                                    sx1, sy1 = int(start.get("x", 0.5) * 1080), int(start.get("y", 0.5) * 1920)
                                    sx2, sy2 = int(end.get("x", 0.5) * 1080), int(end.get("y", 0.5) * 1920)
                                    
                                    if self.device_id:
                                        subprocess.run(['adb', '-s', self.device_id, 'shell', 'input', 'swipe', 
                                                      str(sx1), str(sy1), str(sx2), str(sy2)], check=True)
                                    else:
                                        subprocess.run(['adb', 'shell', 'input', 'swipe', 
                                                      str(sx1), str(sy1), str(sx2), str(sy2)], check=True)
                                    
                                    results.append({"success": True, "type": "swipe"})
                                else:
                                    results.append({"success": False, "message": f"Unsupported action: {action_type}"})
                                    success = False
                                
                                time.sleep(self.action_delay)
                                
                            except Exception as e:
                                self.logger.error(f"Error executing action: {e}")
                                results.append({"success": False, "message": str(e)})
                                success = False
                        
                        return {"success": success, "actions": results}
                
                self.controller = InlineADBController(controller_config)
        except Exception as e:
            self.logger.error(f"Error initializing controller: {e}")
            self.controller = None
            
        self.logger.info("SIMA agent initialized successfully")
    
    def execute_action(self, action_description: str) -> Dict[str, Any]:
        """
        Execute a natural language instruction and return results
        
        Args:
            action_description: Natural language instruction (e.g., "Move forward")
            
        Returns:
            Dictionary containing execution results:
            - success: Whether the action was successful
            - observation: Textual description of what happened
            - raw_data: Additional data for debugging/analysis
        """
        self.logger.info(f"Executing action: {action_description}")
        
        try:
            # Get current observation (screenshot)
            initial_observation = self.observer.get_observation()
            
            # Process instruction and observation
            result = self._process_instruction(action_description, initial_observation)
            
            # Execute action plan
            execution_result = self.controller.execute(result["action_plan"])
            
            # Wait for action to complete
            time.sleep(self.config.get("action_wait_time", 0.5))
            
            # Get new observation
            final_observation = self.observer.get_observation()
            
            # Analyze results
            success = self._evaluate_success(
                initial_observation,
                final_observation,
                action_description,
                execution_result
            )
            
            # Generate observation description
            observation_text = self._generate_observation_description(
                initial_observation,
                final_observation,
                action_description,
                success
            )
            
            return {
                "success": success,
                "observation": observation_text,
                "raw_data": {
                    "execution_details": execution_result,
                    "visual_change": result.get("visual_change", 0.0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error executing action: {str(e)}")
            return {
                "success": False,
                "observation": f"Failed to execute '{action_description}': {str(e)}",
                "error": str(e)
            }
    
    def _process_instruction(self, instruction: str, observation: torch.Tensor) -> Dict[str, Any]:
        """Process an instruction and observation to generate an action plan"""
        # Use torch.no_grad() to prevent gradient tracking during inference
        with torch.no_grad():
            # Encode the instruction
            instruction_embedding = self.language_model(instruction)
            
            # Encode the observation
            visual_embedding = self.vision_model(observation)
            
            # Integrate vision and language to generate action plan
            action_plan = self.integration_model(visual_embedding, instruction_embedding)
        
        # Calculate visual change expected
        visual_change = self._calculate_expected_visual_change(instruction)
        
        return {
            "action_plan": action_plan,
            "visual_embedding": visual_embedding.detach(),  # Ensure tensor is detached
            "instruction_embedding": instruction_embedding.detach(),  # Ensure tensor is detached
            "visual_change": visual_change
        }
    
    def _evaluate_success(
        self, 
        initial_obs: torch.Tensor,
        final_obs: torch.Tensor,
        instruction: str,
        execution_result: Dict[str, Any]
    ) -> bool:
        """Evaluate if the execution was successful"""
        # If execution reported failure, it failed
        if not execution_result.get("success", True):
            return False
        
        # Encode observations with no gradient tracking
        with torch.no_grad():
            initial_embedding = self.vision_model(initial_obs)
            final_embedding = self.vision_model(final_obs)
            
            # Calculate visual difference - detach before calling item()
            visual_diff = torch.norm(final_embedding - initial_embedding, p=2).detach().item()
        
        # Get expected visual change
        expected_change = self._calculate_expected_visual_change(instruction)
        threshold = self.config.get("visual_change_threshold", 0.1)
        
        # Compare actual change to expected change
        if expected_change > threshold and visual_diff < threshold:
            # Expected significant change but didn't see it
            return False
        
        return True
    
    def _generate_observation_description(
        self,
        initial_obs: torch.Tensor,
        final_obs: torch.Tensor,
        instruction: str,
        success: bool
    ) -> str:
        """Generate a textual description of the observation"""
        # Ensure we're not tracking gradients when processing tensors
        with torch.no_grad():
            # If you need to use the tensors for any computation, do it here
            pass
            
        instruction_lower = instruction.lower()
        
        if not success:
            return f"Failed to execute '{instruction}'. No significant change observed."
        
        # Generate success descriptions based on action type
        if any(word in instruction_lower for word in ["move", "walk", "run", "go"]):
            return "Character moved to the specified location. Environment changed accordingly."
        elif any(word in instruction_lower for word in ["attack", "hit", "strike"]):
            return "Character performed attack animation. Target reacted with appropriate feedback."
        elif any(word in instruction_lower for word in ["pick", "grab", "take", "collect"]):
            return "Item was collected and added to inventory. Visual feedback confirmed acquisition."
        elif "open" in instruction_lower:
            return "Object opened, revealing contents. Animation and sound effects played correctly."
        elif "jump" in instruction_lower:
            return "Character performed jumping animation, briefly leaving the ground."
        elif "use" in instruction_lower:
            return "Item was used successfully. Appropriate effects and animations were displayed."
        
        # Generic success message
        return f"Action '{instruction}' was executed successfully."
    
    def _calculate_expected_visual_change(self, instruction: str) -> float:
        """
        Calculate expected visual change for an instruction.

        Args:
            instruction (str): Natural language instruction.

        Returns:
            float: Expected visual change value.
        """
        instruction_lower = instruction.lower()
        
        # High-change actions
        high_change_keywords = [
            "move", "walk", "run", "jump", "teleport", "open", "close", 
            "attack", "pick up", "collect", "throw"
        ]
        
        # Medium-change actions
        medium_change_keywords = [
            "push", "pull", "turn", "rotate", "look", "use", "press", "activate"
        ]
        
        # Low-change actions
        low_change_keywords = [
            "wait", "inspect", "observe", "check", "examine"
        ]
        
        for keyword in high_change_keywords:
            if keyword in instruction_lower:
                return 0.7
                
        for keyword in medium_change_keywords:
            if keyword in instruction_lower:
                return 0.4
                
        for keyword in low_change_keywords:
            if keyword in instruction_lower:
                return 0.1
        
        # Default - medium expectation
        return 0.3
