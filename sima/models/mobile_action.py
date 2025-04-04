# sima/models/mobile_action.py
import torch
import torch.nn as nn
from typing import Dict, Any, List
import logging

class MobileActionModel(nn.Module):
    """Action model for mobile games that generates touch and swipe actions"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize action model with configuration"""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        self.embedding_dim = config.get("embedding_dim", 512)
        self.hidden_dim = config.get("hidden_dim", 1024)
        
        # Define shared layers for action prediction
        self.shared_layers = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        
        # Define action prediction head (explicitly named action_head)
        self.action_head = nn.Linear(self.hidden_dim, 5)  # 5 actions: tap, swipe up/down/left/right
        
        # Define position prediction for tap actions
        self.position_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2)  # x, y coordinates
        )
        
        self.logger.info("Mobile action model initialized successfully")
    
    def forward(self, integrated_embedding: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Generate actions for a puzzle game from integrated embedding
        
        Args:
            integrated_embedding: Integrated vision-language embedding
            
        Returns:
            List of action dictionaries
        """
        # Apply shared layers
        with torch.no_grad():
            features = self.shared_layers(integrated_embedding)
            
            # Get action logits
            action_scores = self.action_head(features)
            action_probs = torch.softmax(action_scores, dim=-1)
            
            # Select the most appropriate action
            action_type_idx = torch.argmax(action_probs, dim=-1).item()
            
            # Define available actions for this game (4 swipe directions + tap)
            actions = ["tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right"]
            chosen_action = actions[action_type_idx % len(actions)]
        
        # Convert to action dictionary
        if chosen_action == "tap":
            # Get tap position
            position = torch.sigmoid(self.position_head(features))
            x, y = position[0].item(), position[1].item()
            
            return [{
                "type": "touch",
                "action": "tap",
                "position": {"x": x, "y": y}
            }]
        elif chosen_action == "swipe_up":
            return [{
                "type": "swipe",
                "action": "swipe",
                "start": {"x": 0.5, "y": 0.7},
                "end": {"x": 0.5, "y": 0.3},
                "duration": 0.3
            }]
        elif chosen_action == "swipe_down":
            return [{
                "type": "swipe",
                "action": "swipe",
                "start": {"x": 0.5, "y": 0.3},
                "end": {"x": 0.5, "y": 0.7},
                "duration": 0.3
            }]
        elif chosen_action == "swipe_left":
            return [{
                "type": "swipe",
                "action": "swipe",
                "start": {"x": 0.7, "y": 0.5},
                "end": {"x": 0.3, "y": 0.5},
                "duration": 0.3
            }]
        elif chosen_action == "swipe_right":
            return [{
                "type": "swipe",
                "action": "swipe",
                "start": {"x": 0.3, "y": 0.5},
                "end": {"x": 0.7, "y": 0.5},
                "duration": 0.3
            }]
