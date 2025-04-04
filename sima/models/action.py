# sima/models/action.py
import torch
import torch.nn as nn
from typing import Dict, Any, List
import logging

class ActionModel(nn.Module):
    """
    Action model for SIMA that generates keyboard/mouse actions from integrated representations.
    Converts high-level understanding into concrete control actions.
    """
    
    def __init__(self, config: Dict[str, Any]):
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
        
        # Add the missing action_head attribute
        self.action_head = nn.Linear(self.hidden_dim, 5)  # 5 actions for mobile game controls
        
        # Define position prediction for tap actions
        self.position_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2)  # x, y coordinates (0 to 1)
        )
        
        self.logger.info("Action model initialized successfully")

    
    def forward(self, integrated_embedding: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Generate actions for a jelly block sliding puzzle game from integrated embedding
        
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
            
            # Define available actions for this game (4 swipe directions)
            actions = ["swipe_up", "swipe_down", "swipe_left", "swipe_right"]
            chosen_action = actions[action_type_idx % len(actions)]
        
        # Convert to action dictionary
        if chosen_action == "swipe_up":
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
        
        # Fallback (shouldn't reach here)
        return [{
            "type": "touch",
            "action": "tap",
            "position": {"x": 0.5, "y": 0.5}
        }]


