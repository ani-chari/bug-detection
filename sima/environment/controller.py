import time
import logging
from typing import Dict, Any, List
import subprocess
import uiautomator2 as u2

class InputController:
    """
    Input controller that executes actions in the Android emulator via ADB
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        self.action_delay = config.get("action_delay", 0.1)
        
        # Initialize connection to the emulator
        try:
            # Get the device serial from ADB
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            device_lines = result.stdout.strip().split('\n')[1:]
            
            if not device_lines:
                self.logger.error("No Android devices/emulators found")
                self.device = None
                return
                
            # Use the first available device
            # device_serial = device_lines[0].split('\t')[0]
            device_serial = "localhost:5555"
            self.device = u2.connect(device_serial)
            self.logger.info(f"Connected to device: {device_serial}")
            
        except Exception as e:
            self.logger.error(f"Error connecting to Android emulator: {str(e)}")
            self.device = None
    
    def execute(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a sequence of actions in the emulator
        
        Args:
            actions: List of action dictionaries
            
        Returns:
            Dictionary containing execution results
        """
        if not actions:
            return {"success": False, "message": "No actions to execute"}
            
        if self.device is None:
            return {"success": False, "message": "No device connected"}
        
        all_results = []
        success = True
        
        for i, action in enumerate(actions):
            try:
                action_type = action.get("type", "")
                
                if action_type == "touch":
                    result = self._execute_touch_action(action)
                elif action_type == "swipe":
                    result = self._execute_swipe_action(action)
                elif action_type == "key":
                    result = self._execute_key_action(action)
                elif action_type == "text":
                    result = self._execute_text_action(action)
                else:
                    result = {"success": False, "message": f"Unknown action type: {action_type}"}
                
                all_results.append(result)
                
                if not result.get("success", False):
                    success = False
                    self.logger.warning(f"Action {i} failed: {result.get('message', 'Unknown error')}")
                
                # Delay between actions
                time.sleep(self.action_delay)
                
            except Exception as e:
                self.logger.error(f"Error executing action {i}: {str(e)}")
                all_results.append({
                    "success": False,
                    "message": str(e),
                    "action": action
                })
                success = False
        
        return {
            "success": success,
            "actions": all_results
        }
    
    def _execute_touch_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a touch action"""
        try:
            action_name = action.get("action", "tap")
            
            if action_name == "tap":
                # Get position
                position = action.get("position", {})
                x = position.get("x", 0.5)
                y = position.get("y", 0.5)
                
                # Convert normalized coordinates to actual screen coordinates
                info = self.device.info
                screen_width, screen_height = info["displayWidth"], info["displayHeight"]
                screen_x = int(x * screen_width)
                screen_y = int(y * screen_height)
                
                # Perform tap
                self.device.click(screen_x, screen_y)
                return {"success": True, "type": "touch", "action": "tap", "position": (screen_x, screen_y)}
                
            elif action_name == "long_press":
                # Get position
                position = action.get("position", {})
                x = position.get("x", 0.5)
                y = position.get("y", 0.5)
                
                # Convert normalized coordinates
                info = self.device.info
                screen_width, screen_height = info["displayWidth"], info["displayHeight"]
                screen_x = int(x * screen_width)
                screen_y = int(y * screen_height)
                
                # Duration
                duration = action.get("duration", 1.0)
                
                # Perform long press
                self.device.long_click(screen_x, screen_y, duration)
                return {"success": True, "type": "touch", "action": "long_press", "position": (screen_x, screen_y)}
                
            else:
                return {"success": False, "message": f"Unknown touch action: {action_name}"}
                
        except Exception as e:
            return {"success": False, "message": str(e), "type": "touch"}
    
    def _execute_swipe_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a swipe action"""
        try:
            # Get start and end positions
            start = action.get("start", {"x": 0.5, "y": 0.7})
            end = action.get("end", {"x": 0.5, "y": 0.3})
            
            # Convert normalized coordinates
            info = self.device.info
            screen_width, screen_height = info["displayWidth"], info["displayHeight"]
            start_x = int(start.get("x", 0.5) * screen_width)
            start_y = int(start.get("y", 0.5) * screen_height)
            end_x = int(end.get("x", 0.5) * screen_width)
            end_y = int(end.get("y", 0.5) * screen_height)
            
            # Duration
            duration = action.get("duration", 0.5)
            
            # Perform swipe
            self.device.swipe(start_x, start_y, end_x, end_y, duration=int(duration * 1000))
            return {
                "success": True, 
                "type": "swipe", 
                "start": (start_x, start_y),
                "end": (end_x, end_y)
            }
                
        except Exception as e:
            return {"success": False, "message": str(e), "type": "swipe"}
    
    def _execute_key_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a key action"""
        try:
            key = action.get("key", "")
            
            if not key:
                return {"success": False, "message": "No key specified"}
            
            # Map common keys
            key_map = {
                "back": "back",
                "home": "home",
                "menu": "menu",
                "power": "power",
                "enter": "enter"
            }
            
            mapped_key = key_map.get(key.lower(), key)
            
            # Press key
            self.device.press(mapped_key)
            return {"success": True, "type": "key", "key": key}
                
        except Exception as e:
            return {"success": False, "message": str(e), "type": "key"}
    
    def _execute_text_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a text input action"""
        try:
            text = action.get("text", "")
            
            if not text:
                return {"success": False, "message": "No text specified"}
            
            # Input text
            self.device.send_keys(text)
            return {"success": True, "type": "text", "text": text}
                
        except Exception as e:
            return {"success": False, "message": str(e), "type": "text"}
