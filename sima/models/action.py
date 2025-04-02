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
        """Initialize action model with configuration"""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        self.embedding_dim = config.get("embedding_dim", 512)
        self.hidden_dim = config.get("hidden_dim", 1024)
        
        # Define keyboard and mouse action spaces
        self.keyboard_keys = config.get("keyboard_keys", [
            "w", "a", "s", "d", "q", "e", "r", "f", "space", "shift", "ctrl",
            "tab", "esc", "enter", "1", "2", "3", "4", "5"
        ])
        
        # Number of actions
        self.num_keyboard_actions = len(self.keyboard_keys) * 2  # Press and release
        self.num_mouse_actions = 4  # Move, left click, right click, scroll
        
        # Action predictor networks
        self.shared_layers = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        
        # Action-specific heads
        self.keyboard_head = nn.Linear(self.hidden_dim, self.num_keyboard_actions)
        self.mouse_action_head = nn.Linear(self.hidden_dim, self.num_mouse_actions)
        self.mouse_position_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2)  # x, y coordinates (0 to 1)
        )
        
        self.logger.info("Action model initialized successfully")
    
    def forward(self, integrated_embedding: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Generate mobile-specific actions for a game with Subway Surfers-like mechanics
        from integrated vision-language embedding.
        
        Args:
            integrated_embedding: Tensor of shape (batch_size, embedding_dim)
                containing the integrated vision-language representation
                
        Returns:
            List of action dictionaries ready for execution by the input controller
        """
        # Extract features using shared layers
        features = self.shared_layers(integrated_embedding)
        
        # Action type prediction
        # Higher score for more appropriate action given the current game state
        action_scores = self.action_head(features)
        action_probs = torch.softmax(action_scores, dim=-1)
        
        # Get the most appropriate action type based on highest probability
        action_type_idx = torch.argmax(action_probs, dim=-1).item()
        action_types = ["tap", "swipe_up", "swipe_down", "swipe_left", "swipe_right"]
        chosen_action = action_types[action_type_idx]
        
        # Position prediction for taps and swipes
        position_coords = torch.sigmoid(self.position_head(features))  # Range [0,1]
        x_pos, y_pos = position_coords[0].item(), position_coords[1].item()
        
        # Create action based on predicted type
        if chosen_action == "tap":
            # Return a tap action at the predicted position
            return [{
                "type": "touch",
                "action": "tap",
                "position": {
                    "x": x_pos,
                    "y": y_pos
                }
            }]
        elif chosen_action == "swipe_up":
            # Swipe up (jump in Subway Surfers)
            return [{
                "type": "swipe",
                "action": "swipe",
                "start": {
                    "x": 0.5,  # Center of screen horizontally
                    "y": 0.7   # Lower part of screen
                },
                "end": {
                    "x": 0.5,  # Keep horizontal position
                    "y": 0.2   # Swipe toward top
                },
                "duration": 0.2  # Fast swipe
            }]
        elif chosen_action == "swipe_down":
            # Swipe down (roll in Subway Surfers)
            return [{
                "type": "swipe",
                "action": "swipe",
                "start": {
                    "x": 0.5,  # Center of screen horizontally
                    "y": 0.3   # Upper part of screen
                },
                "end": {
                    "x": 0.5,  # Keep horizontal position
                    "y": 0.8   # Swipe toward bottom
                },
                "duration": 0.2  # Fast swipe
            }]
        elif chosen_action == "swipe_left":
            # Swipe left (move left in Subway Surfers)
            return [{
                "type": "swipe",
                "action": "swipe",
                "start": {
                    "x": 0.7,  # Right part of screen
                    "y": 0.5   # Center of screen vertically
                },
                "end": {
                    "x": 0.3,  # Swipe toward left
                    "y": 0.5   # Keep vertical position
                },
                "duration": 0.15  # Very fast swipe
            }]
        elif chosen_action == "swipe_right":
            # Swipe right (move right in Subway Surfers)
            return [{
                "type": "swipe",
                "action": "swipe",
                "start": {
                    "x": 0.3,  # Left part of screen
                    "y": 0.5   # Center of screen vertically
                },
                "end": {
                    "x": 0.7,  # Swipe toward right
                    "y": 0.5   # Keep vertical position
                },
                "duration": 0.15  # Very fast swipe
            }]
        
        # Fallback (shouldn't normally reach here)
        return [{
            "type": "touch",
            "action": "tap",
            "position": {
                "x": 0.5,
                "y": 0.5
            }
        }]

