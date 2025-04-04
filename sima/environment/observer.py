import torch
import numpy as np
import logging
from typing import Dict, Any
import subprocess
import tempfile
import os
from PIL import Image

class ScreenObserver:
    """
    Screen observer that captures screenshots from the Android emulator via ADB
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize screen observer with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        self.resize_shape = config.get("resize_shape", (224, 224))
    
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
            img = Image.open(temp_path).convert('RGB')  # Add this explicit conversion to RGB
            
            # Resize image
            img = img.resize(self.resize_shape)
            
            # Convert to numpy array
            img_np = np.array(img)
            
            # Ensure the image has the right dimensions (H, W, 3)
            if len(img_np.shape) == 2:  # Grayscale
                img_np = np.stack([img_np, img_np, img_np], axis=2)
            elif img_np.shape[2] == 4:  # RGBA
                img_np = img_np[:, :, :3]  # Take only RGB channels
                
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
