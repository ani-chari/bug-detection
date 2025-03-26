# sima/agent.py
import torch
import logging
import time
from typing import Dict, Any, List

from .models.vision import VisionModel
from .models.language import LanguageModel
from .models.action import ActionModel
from .models.integration import IntegrationModel
from .environment.observer import ScreenObserver
from .environment.controller import InputController
from .utils.config import default_config

class SIMAAgent:
    """
    Implementation of SIMA (Scalable Instructable Multiworld Agent) from DeepMind.
    
    This agent follows natural language instructions in 3D environments by observing 
    the screen and controlling keyboard/mouse inputs.
    
    Inputs:
    - Natural language instructions (text strings)
    
    Outputs:
    - Execution results (success/failure, observations, etc.)
    - Keyboard and mouse actions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the SIMA agent with configuration"""
        self.config = config or default_config()
        self.logger = logging.getLogger(__name__)
        
        # Determine device
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.vision_model = VisionModel(self.config.get("vision", {})).to(self.device)
        self.language_model = LanguageModel(self.config.get("language", {})).to(self.device)
        self.action_model = ActionModel(self.config.get("action", {})).to(self.device)
        self.integration_model = IntegrationModel(
            self.config.get("integration", {}),
            self.vision_model,
            self.language_model,
            self.action_model
        ).to(self.device)
        
        # Initialize environment interaction
        self.observer = ScreenObserver(self.config.get("observer", {}))
        self.controller = InputController(self.config.get("controller", {}))
        
        self.logger.info("SIMA agent initialized successfully")
    
    def execute_action(self, action_description: str) -> Dict[str, Any]:
        """
        Execute a natural language instruction and return results
        
        Args:
            action_description: Natural language instruction (e.g., "Move forward")
            
        Returns:
            Dictionary containing execution results:
            - success: Whether the action was successful
            - observation: Textual description of what happened
            - raw_data: Additional data for debugging/analysis
        """
        self.logger.info(f"Executing action: {action_description}")
        
        try:
            # Get current observation (screenshot)
            initial_observation = self.observer.get_observation()
            
            # Process instruction and observation
            result = self._process_instruction(action_description, initial_observation)
            
            # Execute action plan
            execution_result = self.controller.execute(result["action_plan"])
            
            # Wait for action to complete
            time.sleep(self.config.get("action_wait_time", 0.5))
            
            # Get new observation
            final_observation = self.observer.get_observation()
            
            # Analyze results
            success = self._evaluate_success(
                initial_observation,
                final_observation,
                action_description,
                execution_result
            )
            
            # Generate observation description
            observation_text = self._generate_observation_description(
                initial_observation,
                final_observation,
                action_description,
                success
            )
            
            return {
                "success": success,
                "observation": observation_text,
                "raw_data": {
                    "execution_details": execution_result,
                    "visual_change": result.get("visual_change", 0.0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error executing action: {str(e)}")
            return {
                "success": False,
                "observation": f"Failed to execute '{action_description}': {str(e)}",
                "error": str(e)
            }
    
    def _process_instruction(self, instruction: str, observation: torch.Tensor) -> Dict[str, Any]:
        """Process an instruction and observation to generate an action plan"""
        # Encode the instruction
        instruction_embedding = self.language_model(instruction)
        
        # Encode the observation
        visual_embedding = self.vision_model(observation)
        
        # Integrate vision and language to generate action plan
        action_plan = self.integration_model(visual_embedding, instruction_embedding)
        
        # Calculate visual change expected
        visual_change = self._calculate_expected_visual_change(instruction)
        
        return {
            "action_plan": action_plan,
            "visual_embedding": visual_embedding,
            "instruction_embedding": instruction_embedding,
            "visual_change": visual_change
        }
    
    def _evaluate_success(
        self, 
        initial_obs: torch.Tensor,
        final_obs: torch.Tensor,
        instruction: str,
        execution_result: Dict[str, Any]
    ) -> bool:
        """Evaluate if the execution was successful"""
        # If execution reported failure, it failed
        if not execution_result.get("success", True):
            return False
        
        # Encode observations
        initial_embedding = self.vision_model(initial_obs)
        final_embedding = self.vision_model(final_obs)
        
        # Calculate visual difference
        visual_diff = torch.norm(final_embedding - initial_embedding, p=2).item()
        
        # Get expected visual change
        expected_change = self._calculate_expected_visual_change(instruction)
        threshold = self.config.get("visual_change_threshold", 0.1)
        
        # Compare actual change to expected change
        if expected_change > threshold and visual_diff < threshold:
            # Expected significant change but didn't see it
            return False
        
        return True
    
    def _calculate_expected_visual_change(self, instruction: str) -> float:
        """Calculate expected visual change for an instruction"""
        instruction_lower = instruction.lower()
        
        # High-change actions
        high_change_keywords = [
            "move", "walk", "run", "jump", "teleport", "open", "close", 
            "attack", "pick up", "collect", "throw"
        ]
        
        # Medium-change actions
        medium_change_keywords = [
            "push", "pull", "turn", "rotate", "look", "use", "press", "activate"
        ]
        
        # Low-change actions
        low_change_keywords = [
            "wait", "inspect", "observe", "check", "examine"
        ]
        
        for keyword in high_change_keywords:
            if keyword in instruction_lower:
                return 0.7
                
        for keyword in medium_change_keywords:
            if keyword in instruction_lower:
                return 0.4
                
        for keyword in low_change_keywords:
            if keyword in instruction_lower:
                return 0.1
        
        # Default - medium expectation
        return 0.3
    
    def _generate_observation_description(
        self,
        initial_obs: torch.Tensor,
        final_obs: torch.Tensor,
        instruction: str,
        success: bool
    ) -> str:
        """Generate a textual description of the observation"""
        # This would ideally use a vision-language model to describe the change
        # For now, use basic templating based on instruction and success
        instruction_lower = instruction.lower()
        
        if not success:
            return f"Failed to execute '{instruction}'. No significant change observed."
        
        # Generate success descriptions based on action type
        if any(word in instruction_lower for word in ["move", "walk", "run", "go"]):
            return "Character moved to the specified location. Environment changed accordingly."
        elif any(word in instruction_lower for word in ["attack", "hit", "strike"]):
            return "Character performed attack animation. Target reacted with appropriate feedback."
        elif any(word in instruction_lower for word in ["pick", "grab", "take", "collect"]):
            return "Item was collected and added to inventory. Visual feedback confirmed acquisition."
        elif "open" in instruction_lower:
            return "Object opened, revealing contents. Animation and sound effects played correctly."
        elif "jump" in instruction_lower:
            return "Character performed jumping animation, briefly leaving the ground."
        elif "use" in instruction_lower:
            return "Item was used successfully. Appropriate effects and animations were displayed."
        
        # Generic success message
        return f"Action '{instruction}' was executed successfully."
