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
        Generate actions from integrated embedding
        
        Args:
            integrated_embedding: Integrated vision-language embedding
            
        Returns:
            List of action dictionaries
        """
        # Apply shared layers
        features = self.shared_layers(integrated_embedding)
        
        # Get action logits
        keyboard_logits = self.keyboard_head(features)
        mouse_action_logits = self.mouse_action_head(features)
        mouse_position = torch.sigmoid(self.mouse_position_head(features))  # 0 to 1
        
        # Convert to probabilities
        keyboard_probs = torch.softmax(keyboard_logits, dim=-1)
        mouse_action_probs = torch.softmax(mouse_action_logits, dim=-1)
        
        # Select top actions
        keyboard_action = torch.argmax(keyboard_probs, dim=-1).item()
        mouse_action = torch.argmax(mouse_action_probs, dim=-1).item()
        
        # Create action plan
        action_plan = []
        
        # Add keyboard action
        if keyboard_action < self.num_keyboard_actions:
            key_idx = keyboard_action // 2
            is_press = keyboard_action % 2 == 0
            key = self.keyboard_keys[key_idx]
            
            action_plan.append({
                "type": "keyboard",
                "key": key,
                "action": "press" if is_press else "release"
            })
        
        # Add mouse action
        mouse_actions = ["move", "left_click", "right_click", "scroll"]
        if mouse_action < len(mouse_actions):
            mouse_action_name = mouse_actions[mouse_action]
            
            if mouse_action_name == "move":
                # Get position coordinates
                x, y = mouse_position[0].cpu().numpy().tolist()
                
                action_plan.append({
                    "type": "mouse",
                    "action": "move",
                    "position": {"x": x, "y": y}
                })
            else:
                action_plan.append({
                    "type": "mouse",
                    "action": mouse_action_name
                })
        
        return action_plan
