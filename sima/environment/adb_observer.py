# sima/environment/adb_observer.py
import subprocess
import tempfile
import os
from PIL import Image
import numpy as np
import torch
import logging
from typing import Dict, Any

class ADBScreenObserver:
    """Screen observer that captures screenshots from Android devices via ADB"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize observer with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.resize_shape = config.get("resize_shape", (224, 224))
        
        # Initialize device ID
        self.device_id = config.get("device_id", None)
        if not self.device_id:
            self._find_device_id()
            
    def _find_device_id(self):
        """Find a connected BlueStacks device ID"""
        try:
            result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')[1:]  # Skip the first line
            
            bluestacks_devices = []
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2 and "device" in parts[1]:
                        device_id = parts[0]
                        # Check if this is BlueStacks (usually has localhost in the name)
                        if "localhost" in device_id:
                            bluestacks_devices.append(device_id)
            
            if bluestacks_devices:
                self.device_id = bluestacks_devices[0]
                self.logger.info(f"Using BlueStacks device: {self.device_id}")
            elif lines:
                # Use the first available device if no BlueStacks detected
                self.device_id = lines[0].split()[0]
                self.logger.info(f"Using device: {self.device_id}")
            else:
                self.logger.warning("No devices found. ADB commands will likely fail.")
                self.device_id = None  # Ensure it's explicitly None
        except Exception as e:
            self.logger.error(f"Error finding device ID: {e}")
            self.device_id = None  # Ensure it's explicitly None
    
    def get_observation(self) -> torch.Tensor:
        """Capture and process a screenshot from the emulator"""
        try:
            # Create a temporary file for the screenshot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Use ADB to capture screenshot with specific device ID
            if self.device_id:
                subprocess.run(['adb', '-s', self.device_id, 'shell', 'screencap', '-p', '/sdcard/screenshot.png'], check=True)
                subprocess.run(['adb', '-s', self.device_id, 'pull', '/sdcard/screenshot.png', temp_path], check=True)
            else:
                self.logger.warning("No device ID available, attempting without -s flag")
                subprocess.run(['adb', 'shell', 'screencap', '-p', '/sdcard/screenshot.png'], check=True)
                subprocess.run(['adb', 'pull', '/sdcard/screenshot.png', temp_path], check=True)
            
            # Load the image and explicitly convert to RGB
            img = Image.open(temp_path).convert('RGB')
            
            # Resize image
            img = img.resize(self.resize_shape)
            
            # Convert to numpy array
            img_np = np.array(img)
            
            # Convert to tensor (C, H, W) - PyTorch format
            img_tensor = torch.tensor(img_np).permute(2, 0, 1).float() / 255.0
            
            # Clean up
            os.unlink(temp_path)
            
            return img_tensor
            
        except Exception as e:
            self.logger.error(f"Error capturing emulator screen: {e}")
            # Return a blank image on error
            blank = np.zeros((self.resize_shape[1], self.resize_shape[0], 3), dtype=np.uint8)
            return torch.tensor(blank).permute(2, 0, 1).float() / 255.0
