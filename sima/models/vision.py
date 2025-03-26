# sima/models/vision.py
import torch
import torch.nn as nn
from typing import Dict, Any
import logging
from transformers import AutoModel, AutoProcessor

class VisionModel(nn.Module):
    """
    Vision model component for SIMA that processes visual observations.
    Follows the architecture described in the SIMA paper using pre-trained
    vision transformers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize vision model with configuration"""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        self.model_name = config.get("model_name", "openai/clip-vit-base-patch16")
        
        try:
            # Load pre-trained vision model
            self.logger.info(f"Loading vision model: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Get embedding dimension
            self.embedding_dim = self.model.config.hidden_size
            
            # Add projection layer (as mentioned in SIMA paper)
            self.projection = nn.Sequential(
                nn.Linear(self.embedding_dim, 1024),
                nn.GELU(),
                nn.Linear(1024, 512)
            )
            
            self.logger.info("Vision model initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading vision model: {str(e)}")
            # Fallback to a simple CNN if transformer loading fails
            self.model = self._create_fallback_model()
            self.processor = None
            self.embedding_dim = 512
            self.projection = nn.Identity()
    
    def _create_fallback_model(self) -> nn.Module:
        """Create a fallback CNN model if transformer loading fails"""
        self.logger.info("Creating fallback CNN model")
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU()
        )
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Process visual observation and return embedding
        
        Args:
            observation: Image tensor of shape (B, C, H, W)
            
        Returns:
            Visual embedding tensor
        """
        with torch.no_grad():
            if self.processor is not None:
                # Process with transformer
                if observation.dim() == 3:
                    observation = observation.unsqueeze(0)  # Add batch dimension
                
                inputs = self.processor(images=observation, return_tensors="pt").to(observation.device)
                outputs = self.model.get_image_features(**inputs)
                
                if outputs.dim() > 2:
                    # Use mean pooling if needed
                    outputs = outputs.mean(dim=1)
            else:
                # Process with fallback CNN
                outputs = self.model(observation)
        
        # Project to final dimension
        embedding = self.projection(outputs)
        return embedding
