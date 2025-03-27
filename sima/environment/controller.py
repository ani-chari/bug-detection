# sima/environment/controller.py
import time
import logging
from typing import Dict, Any, List
import pyautogui

class InputController:
    """
    Input controller that executes keyboard and mouse actions in the game.
    This component provides action execution capabilities to the SIMA agent.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize input controller with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        self.action_delay = config.get("action_delay", 0.1)
        
        # Configure PyAutoGUI
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.05
        
        self.logger.info("Input controller initialized successfully")
    
    def execute(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a sequence of actions
        
        Args:
            actions: List of action dictionaries
            
        Returns:
            Dictionary containing execution results
        """
        if not actions:
            return {"success": False, "message": "No actions to execute"}
        
        all_results = []
        success = True
        
        for i, action in enumerate(actions):
            try:
                action_type = action.get("type", "")
                
                if action_type == "keyboard":
                    result = self._execute_keyboard_action(action)
                elif action_type == "mouse":
                    result = self._execute_mouse_action(action)
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
    
    def _execute_keyboard_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a keyboard action"""
        key = action.get("key", "")
        action_name = action.get("action", "press")
        
        if not key:
            return {"success": False, "message": "No key specified"}
        
        try:
            if action_name == "press":
                pyautogui.keyDown(key)
                return {"success": True, "type": "keyboard", "key": key, "action": "press"}
            elif action_name == "release":
                pyautogui.keyUp(key)
                return {"success": True, "type": "keyboard", "key": key, "action": "release"}
            else:
                return {"success": False, "message": f"Invalid keyboard action: {action_name}"}
        except Exception as e:
            return {"success": False, "message": str(e), "type": "keyboard", "key": key}
    
    def _execute_mouse_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a mouse action"""
        action_name = action.get("action", "")
        
        if not action_name:
            return {"success": False, "message": "No mouse action specified"}
        
        try:
            if action_name == "move":
                # Get position
                position = action.get("position", {})
                x = position.get("x", 0.5)
                y = position.get("y", 0.5)
                
                # Convert normalized coordinates to screen coordinates
                screen_width, screen_height = pyautogui.size()
                screen_x = int(x * screen_width)
                screen_y = int(y * screen_height)
                
                # Move mouse
                pyautogui.moveTo(screen_x, screen_y)
                return {"success": True, "type": "mouse", "action": "move", "position": (screen_x, screen_y)}
                
            elif action_name == "left_click":
                pyautogui.click()
                return {"success": True, "type": "mouse", "action": "left_click"}
                
            elif action_name == "right_click":
                pyautogui.rightClick()
                return {"success": True, "type": "mouse", "action": "right_click"}
                
            elif action_name == "scroll":
                # Get scroll amount
                amount = action.get("amount", 5)
                pyautogui.scroll(amount)
                return {"success": True, "type": "mouse", "action": "scroll", "amount": amount}
                
            else:
                return {"success": False, "message": f"Invalid mouse action: {action_name}"}
                
        except Exception as e:
            return {"success": False, "message": str(e), "type": "mouse", "action": action_name}
