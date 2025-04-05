import subprocess
import time
import logging
from typing import Dict, Any, List

class ADBInputController:
    """Controller that sends input commands to Android devices via ADB"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize controller with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device_id = config.get("device_id", None)
        self.action_delay = config.get("action_delay", 0.5)
        
        # Initialize device ID if not provided
        if not self.device_id:
            self._find_device_id()
            
        # Initialize uiautomator2
        self.u2 = None
        try:
            import uiautomator2 as u2
            if self.device_id:
                self.u2 = u2.connect(self.device_id)
            else:
                self.u2 = u2.connect()
            self.logger.info("uiautomator2 initialized successfully")
            self.logger.info(f"Connected to device: {self.device_id}")
        except ImportError:
            self.logger.warning("uiautomator2 not installed. Falling back to ADB input.")
        except Exception as e:
            self.logger.warning(f"Failed to connect via uiautomator2: {e}. Falling back to ADB input.")
        
    def _find_device_id(self):
        """Find a connected Android device ID"""
        try:
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')[1:]
            
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2 and "device" in parts[1]:
                        self.device_id = parts[0]
                        break
            
            if self.device_id:
                self.logger.info(f"Using device: {self.device_id}")
            else:
                self.logger.warning("No devices found. ADB commands may fail.")
        except Exception as e:
            self.logger.error(f"Error finding device ID: {e}")
    
    def execute(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a list of actions"""
        results = []
        success = True
        
        for action in actions:
            action_type = action.get("type", "")
            
            try:
                if action_type == "touch" and action.get("action") == "tap":
                    pos = action.get("position", {"x": 0.5, "y": 0.5})
                    x, y = pos.get("x", 0.5), pos.get("y", 0.5)
                    
                    if self.u2:
                        try:
                            # Use uiautomator2 for tapping
                            self.u2.click(x, y)
                            results.append({"success": True, "type": "touch", "action": "tap", "via": "uiautomator2"})
                        except Exception as e:
                            self.logger.warning(f"uiautomator2 tap failed: {e}. Falling back to ADB input.")
                            self.u2 = None  # Disable uiautomator2 if it consistently fails
                            # Fallback to ADB input
                            sx, sy = int(x * self.u2.window_size()[0]), int(y * self.u2.window_size()[1])
                            subprocess.run(['adb', '-s', self.device_id, 'shell', 'input', 'tap', str(sx), str(sy)], check=True, timeout=30)
                            results.append({"success": True, "type": "touch", "action": "tap", "via": "adb"})
                    else:
                        # Use ADB input if uiautomator2 is not available
                        sx, sy = int(x * 1080), int(y * 1920)  # Hardcoded screen size
                        subprocess.run(['adb', '-s', self.device_id, 'shell', 'input', 'tap', str(sx), str(sy)], check=True, timeout=30)
                        results.append({"success": True, "type": "touch", "action": "tap", "via": "adb"})
                
                elif action_type == "swipe":
                    start = action.get("start", {"x": 0.5, "y": 0.7})
                    end = action.get("end", {"x": 0.5, "y": 0.3})
                    x1, y1 = start.get("x", 0.5), start.get("y", 0.7)
                    x2, y2 = end.get("x", 0.5), end.get("y", 0.3)
                    
                    if self.u2:
                        try:
                            # Use uiautomator2 for swiping
                            self.u2.swipe(x1, y1, x2, y2)
                            results.append({"success": True, "type": "swipe", "via": "uiautomator2"})
                        except Exception as e:
                            self.logger.warning(f"uiautomator2 swipe failed: {e}. Falling back to ADB input.")
                            self.u2 = None  # Disable uiautomator2 if it consistently fails
                            # Fallback to ADB input
                            sx1, sy1 = int(x1 * self.u2.window_size()[0]), int(y1 * self.u2.window_size()[1])
                            sx2, sy2 = int(x2 * self.u2.window_size()[0]), int(y2 * self.u2.window_size()[1])
                            subprocess.run(['adb', '-s', self.device_id, 'shell', 'input', 'swipe', str(sx1), str(sy1), str(sx2), str(sy2)], check=True, timeout=30)
                            results.append({"success": True, "type": "swipe", "via": "adb"})
                    else:
                        # Use ADB input if uiautomator2 is not available
                        sx1, sy1 = int(x1 * 1080), int(y1 * 1920)  # Hardcoded screen size
                        sx2, sy2 = int(x2 * 1080), int(y2 * 1920)  # Hardcoded screen size
                        subprocess.run(['adb', '-s', self.device_id, 'shell', 'input', 'swipe', str(sx1), str(sy1), str(sx2), str(sy2)], check=True, timeout=30)
                        results.append({"success": True, "type": "swipe", "via": "adb"})
                
                else:
                    results.append({"success": False, "message": f"Unsupported action: {action_type}"})
                    success = False
                
                time.sleep(self.action_delay)
                
            except subprocess.TimeoutExpired:
                self.logger.error("ADB command timed out")
                results.append({"success": False, "message": "ADB command timed out"})
                success = False
            except Exception as e:
                self.logger.error(f"Error executing action: {e}")
                results.append({"success": False, "message": str(e)})
                success = False
        
        return {"success": success, "actions": results}
    

class SimpleGameController:
    def __init__(self, config):
        self.config = config
        self.device_id = config.get("device_id")
        self.logger = logging.getLogger(__name__)
        self.action_delay = config.get("action_delay", 0.5)
        
    def execute(self, actions):
        """Execute a list of actions"""
        if not isinstance(actions, list):
            actions = [actions]  # Convert single action to list
            
        results = []
        success = True
        
        for action in actions:
            try:
                action_type = action.get("type", "")
                
                if action_type == "swipe":
                    start = action.get("start", {})
                    end = action.get("end", {})
                    self._execute_swipe(
                        start_x=start.get("x", 0.5), 
                        start_y=start.get("y", 0.5),
                        end_x=end.get("x", 0.5), 
                        end_y=end.get("y", 0.5)
                    )
                    results.append({"success": True, "type": "swipe"})
                    
                elif action_type == "touch" and action.get("action") == "tap":
                    pos = action.get("position", {})
                    self._execute_tap(x=pos.get("x", 0.5), y=pos.get("y", 0.5))
                    results.append({"success": True, "type": "touch", "action": "tap"})
                    
                else:
                    results.append({"success": False, "message": f"Unsupported action: {action_type}"})
                    success = False
                
                time.sleep(self.action_delay)
                
            except Exception as e:
                self.logger.error(f"Error executing action: {e}")
                results.append({"success": False, "message": str(e)})
                success = False
        
        return {"success": success, "actions": results}
    
    def _execute_swipe(self, start_x, start_y, end_x, end_y):
        """Execute a swipe action using ADB"""
        print(start_x, start_y, end_x, end_y)
        sx1, sy1 = int(start_x * 1080), int(start_y * 1920)
        sx2, sy2 = int(end_x * 1080), int(end_y * 1920)
        
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'swipe', str(sx1), str(sy1), str(sx2), str(sy2)])
        
        subprocess.run(cmd, check=True)
    
    def _execute_tap(self, x, y):
        """Execute a tap action using ADB"""
        sx, sy = int(x * 1080), int(y * 1920)
        
        cmd = ['adb']
        if self.device_id:
            cmd.extend(['-s', self.device_id])
        cmd.extend(['shell', 'input', 'tap', str(sx), str(sy)])
        
        subprocess.run(cmd, check=True)
