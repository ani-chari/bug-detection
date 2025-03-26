# sima/models/integration.py
import torch
import torch.nn as nn
from typing import Dict, Any, List
import logging

class IntegrationModel(nn.Module):
    """
    Integration model for SIMA that combines vision and language representations.
    This follows the multimodal transformer architecture described in the paper.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        vision_model: nn.Module,
        language_model: nn.Module,
        action_model: nn.Module
    ):
        """Initialize integration model with configuration and component models"""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Store component models
        self.vision_model = vision_model
        self.language_model = language_model
        self.action_model = action_model
        
        # Get configuration
        self.embedding_dim = config.get("embedding_dim", 512)
        self.num_heads = config.get("num_heads", 8)
        
        # Integration components
        self.vision_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.language_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        # Cross-attention for vision-language integration (as described in SIMA paper)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            batch_first=True
        )
        
        # Integration layers
        self.integration_layers = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        self.logger.info("Integration model initialized successfully")
    
    def forward(
        self, 
        visual_embedding: torch.Tensor, 
        language_embedding: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """
        Integrate vision and language to generate actions
        
        Args:
            visual_embedding: Visual embedding from vision model
            language_embedding: Language embedding from language model
            
        Returns:
            List of action dictionaries
        """
        # Project the embeddings
        visual_proj = self.vision_projection(visual_embedding)
        language_proj = self.language_projection(language_embedding)
        
        # Reshape for cross-attention
        # Ensure proper dimensions (batch, sequence, features)
        if visual_proj.dim() == 2:
            visual_proj = visual_proj.unsqueeze(1)
        if language_proj.dim() == 2:
            language_proj = language_proj.unsqueeze(1)
        
        # Apply cross-attention (vision attends to language)
        attended_visual, _ = self.cross_attention(
            query=visual_proj,
            key=language_proj,
            value=language_proj
        )
        
        # Concatenate attended visual with original visual 
        if attended_visual.dim() == 3:
            attended_visual = attended_visual.squeeze(1)
        if visual_embedding.dim() == 3:
            visual_embedding = visual_embedding.squeeze(1)
            
        concat_embedding = torch.cat([attended_visual, visual_embedding], dim=1)
        
        # Integrate embeddings
        integrated_embedding = self.integration_layers(concat_embedding)
        
        # Generate actions using action model
        action_plan = self.action_model(integrated_embedding)
        
        return action_plan
