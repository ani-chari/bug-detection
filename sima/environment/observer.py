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
        """
        Capture and process a screenshot from the emulator
        
        Returns:
            Tensor representation of the screenshot (C, H, W)
        """
        try:
            # Create a temporary file for the screenshot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Use ADB to capture screenshot
            subprocess.run(['adb', 'shell', 'screencap', '-p', '/sdcard/screenshot.png'], check=True)
            subprocess.run(['adb', 'pull', '/sdcard/screenshot.png', temp_path], check=True)
            
            # Load the image
            img = Image.open(temp_path)
            
            # Resize image
            img = img.resize(self.resize_shape)
            
            # Convert to numpy array
            img_np = np.array(img)
            
            # Convert to tensor (C, H, W)
            img_tensor = torch.tensor(img_np).permute(2, 0, 1).float() / 255.0
            
            # Clean up
            os.unlink(temp_path)
            
            return img_tensor
            
        except Exception as e:
            self.logger.error(f"Error capturing emulator screen: {str(e)}")
            # Return a blank image on error
            blank = np.zeros((self.resize_shape[1], self.resize_shape[0], 3), dtype=np.uint8)
            return torch.tensor(blank).permute(2, 0, 1).float() / 255.0
