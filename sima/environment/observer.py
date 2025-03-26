# sima/environment/observer.py
import torch
import numpy as np
import logging
from typing import Dict, Any
import mss
import cv2
from PIL import Image

class ScreenObserver:
    """
    Screen observer that captures and processes game screenshots.
    This component provides visual input to the SIMA agent.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize screen observer with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        self.capture_region = config.get("capture_region", None)
        self.resize_shape = config.get("resize_shape", (224, 224))
        
        # Initialize screen capture
        try:
            self.sct = mss.mss()
            self.logger.info("Screen observer initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing screen capture: {str(e)}")
            self.sct = None
    
    def get_observation(self) -> torch.Tensor:
        """
        Capture and process a screenshot
        
        Returns:
            Tensor representation of the screenshot (C, H, W)
        """
        if self.sct is None:
            # Return a blank image if screen capture is not available
            blank = np.zeros((self.resize_shape[1], self.resize_shape[0], 3), dtype=np.uint8)
            return torch.tensor(blank).permute(2, 0, 1).float() / 255.0
        
        try:
            # Capture screen region
            if self.capture_region:
                monitor = self.capture_region
            else:
                monitor = self.sct.monitors[0]
            
            sct_img = self.sct.grab(monitor)
            
            # Convert to numpy array
            img = np.array(sct_img)
            
            # Convert from BGRA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            
            # Resize image
            img = cv2.resize(img, self.resize_shape)
            
            # Convert to tensor (C, H, W)
            img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0
            
            return img_tensor
            
        except Exception as e:
            self.logger.error(f"Error capturing screen: {str(e)}")
            # Return a blank image on error
            blank = np.zeros((self.resize_shape[1], self.resize_shape[0], 3), dtype=np.uint8)
            return torch.tensor(blank).permute(2, 0, 1).float() / 255.0
