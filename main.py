import os
import json
import logging
import time
import uuid
import io
import base64
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import required libraries
import torch
import numpy as np
from PIL import Image
from openai import OpenAI

# Import video processing
from sima.models.video_processor import VideoGameStateProcessor, UnityVideoObserver

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data structures for the pipeline
@dataclass
class TestTask:
    """Test task to be executed by the pipeline"""
    task_id: str
    description: str
    initial_state: str
    expected_outcome: str
    priority: str = "medium"
    potential_bugs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "initial_state": self.initial_state,
            "expected_outcome": self.expected_outcome,
            "priority": self.priority,
            "potential_bugs": self.potential_bugs
        }

@dataclass
class ActionStep:
    """Step in an action plan"""
    step_id: str
    action_description: str
    expected_observation: str
    success_criteria: str
    fallback_action: Optional[str] = None
    is_checkpoint: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action_description": self.action_description,
            "expected_observation": self.expected_observation,
            "success_criteria": self.success_criteria,
            "fallback_action": self.fallback_action,
            "is_checkpoint": self.is_checkpoint
        }

@dataclass
class ExecutionResult:
    """Result of executing an action step"""
    step_id: str
    action_description: str
    observation: str
    success: bool
    raw_data: Dict[str, Any] = field(default_factory=dict)
    is_fallback: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action_description": self.action_description,
            "observation": self.observation,
            "success": self.success,
            "raw_data": self.raw_data,
            "is_fallback": self.is_fallback
        }

@dataclass
class Bug:
    """A detected bug or issue"""
    bug_id: str
    description: str
    severity: str
    reproduction_steps: str
    expected_behavior: str
    actual_behavior: str
    affected_tasks: List[str] = field(default_factory=list)
    potential_fix: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bug_id": self.bug_id,
            "description": self.description,
            "severity": self.severity,
            "reproduction_steps": self.reproduction_steps,
            "expected_behavior": self.expected_behavior,
            "actual_behavior": self.actual_behavior,
            "affected_tasks": self.affected_tasks,
            "potential_fix": self.potential_fix
        }


# Updated SIMAAgent integration for your pipeline
from sima.agent import SIMAAgent as SIMACore
class SIMAAgent(SIMACore):
    """
    SIMA agent integration for the game testing pipeline
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize SIMA agent"""
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + ".SIMAAgent")
        
        # Initialize OpenAI client for enhanced descriptions
        openai_api_key = self.config.get("openai_api_key", os.environ.get("OPENAI_API_KEY"))
        self.llm_client = OpenAI(api_key=openai_api_key)
        
        try:
            # Import our SIMA implementation
            from sima.agent import SIMAAgent as SIMACore
            print(SIMACore.__module__)
            from sima.utils.config import default_config
            
            # Create SIMA configuration by combining defaults with user config
            sima_config = default_config()
            
            # Update with user config
            if "sima" in self.config:
                self._update_config_recursive(sima_config, self.config["sima"])
            
            # Add Unity executable path if provided
            if "unity_executable_path" in self.config:
                if "unity" not in sima_config:
                    sima_config["unity"] = {}
                sima_config["unity"]["executable_path"] = self.config["unity_executable_path"]
            
            # Initialize SIMA core
            self.logger.info("Initializing SIMA core")
            self.sima = SIMACore()  # Use the SIMACore class directly
            self.logger.info("SIMA core initialized successfully")
        except ImportError as e:
            self.logger.error(f"Failed to import SIMA: {str(e)}")
            self.logger.info("Make sure the SIMA package is installed")
            self.logger.info("Using simulated functionality")
            self.sima = None
    
    def _update_config_recursive(self, base_config: Dict[str, Any], update_config: Dict[str, Any]) -> None:
        """Update a configuration dictionary recursively"""
        for key, value in update_config.items():
            if isinstance(value, dict) and key in base_config and isinstance(base_config[key], dict):
                self._update_config_recursive(base_config[key], value)
            else:
                base_config[key] = value
    
    def execute_action(self, action_description: str) -> Dict[str, Any]:
        """Execute a natural language action"""
        self.logger.info(f"Executing action: {action_description}")
        
        if self.sima is None:
            return self._simulated_execution(action_description)
        
        try:
            # Execute the action using SIMA
            result = self.sima.execute_action(action_description)
            
            # Enhance the observation description using an LLM
            enhanced_observation = self._enhance_observation_description(
                action_description,
                result.get("observation", ""),
                result.get("success", False)
            )
            
            # Update the result with the enhanced observation
            result["observation"] = enhanced_observation
            
            return result
            
        except Exception as e:
            self.logger.error(f"SIMA execution error: {str(e)}")
            self.logger.info("Falling back to simulated execution")
            return self._simulated_execution(action_description)
    
    def _enhance_observation_description(
        self,
        action_description: str,
        base_observation: str,
        success: bool
    ) -> str:
        """Enhance observation description using LLM"""
        try:
            prompt = f"""
            Describe what happened after executing this action in a game:
            
            Action: {action_description}
            Base observation: {base_observation}
            Success: {"Yes" if success else "No"}
            
            Provide a concise but detailed description of what was observed.
            """
            
            response = self.llm_client.chat.completions.create(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=150
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Failed to enhance observation description: {e}")
            return base_observation
    
    def _simulated_execution(self, action_description: str) -> Dict[str, Any]:
        """Simulated action execution (fallback)"""
        self.logger.info(f"Using simulated execution for: {action_description}")
        
        # Determine if action should succeed
        success = True
        
        # Check for negative indicators
        negative_indicators = ["impossible", "fail", "can't", "cannot", "unable"]
        if any(indicator in action_description.lower() for indicator in negative_indicators):
            success = False
        
        # Generate appropriate observation based on action type
        action_lower = action_description.lower()
        
        if "move" in action_lower or "walk" in action_lower or "go" in action_lower:
            observation = "Character moved to the specified location. Surrounding environment updated accordingly."
        elif "attack" in action_lower or "hit" in action_lower or "fight" in action_lower:
            observation = "Attack animation played. Target reacted with appropriate feedback and health reduction."
        elif "pick" in action_lower or "grab" in action_lower or "take" in action_lower:
            observation = "Item was collected and added to inventory. Visual and sound feedback confirmed acquisition."
        elif "open" in action_lower:
            observation = "Container/door opened with appropriate animation. Interior/next area now visible."
        elif "use" in action_lower:
            observation = "Item used with expected effect. Animation and particle effects displayed correctly."
        elif "jump" in action_lower:
            observation = "Character performed jumping animation and cleared the obstacle."
        elif "talk" in action_lower or "speak" in action_lower:
            observation = "Dialogue interface appeared with NPC response options."
        elif "teleport" in action_lower or "portal" in action_lower:
            observation = "Character teleported to the target location with appropriate visual effects."
        else:
            observation = f"Action '{action_description}' executed successfully with appropriate visual feedback."
        
        if not success:
            observation = f"Failed to execute '{action_description}'. No significant change observed in the environment."
        
        # Enhance the observation if possible
        try:
            enhanced_observation = self._enhance_observation_description(
                action_description,
                observation,
                success
            )
            observation = enhanced_observation
        except:
            pass
        
        return {
            "success": success,
            "observation": observation,
            "raw_data": {
                "execution_details": {
                    "actions": [{"type": "simulated", "success": success}]
                }
            }
        }



class TesterAgent:
    """
    Agent that generates comprehensive test tasks for a game feature
    """
    def __init__(self, llm_client: OpenAI):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__ + ".TesterAgent")
    
    def generate_test_tasks(self, game_description: str, feature_description: str, num_tasks: int = 10):
        """Generate general game testing tasks with mechanics-focused approach"""
        logger.info("Tester agent: Generating test tasks")
        
        prompt = f"""
        You are an expert game tester who creates comprehensive test plans for any type of video game.
        Generate {num_tasks} test tasks that focus on understanding game mechanics and identifying potential issues.
        
        Game Description:
        {game_description}
        
        Feature to Test:
        {feature_description}
        
        Create test tasks that:
        1. Explore core game mechanics in depth
        2. Test boundary conditions where rules might break
        3. Verify that game objectives are achievable
        4. Check for consistency in how rules are applied
        5. Examine edge cases where mechanics interact in unexpected ways
        
        Each test should be designed to first understand how the game works before assuming anything is a bug.
        Remember that unsuccessful actions may be valid game constraints, not bugs.
        
        For each test case, provide:
        - task_id: A unique identifier (e.g., "TASK-001")
        - description: Detailed description of what aspect of the game to test
        - initial_state: What condition the game should be in before testing
        - expected_outcome: What should happen if the game is functioning correctly
        - potential_bugs: What types of issues this test might reveal
        
        Format your response as a JSON object with an array of test task objects under the "tasks" key.
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are a specialized AI for generating game testing tasks"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            task_dicts = result.get("tasks", [])
            
            # Convert dictionaries to TestTask objects
            test_tasks = []
            for i, task_dict in enumerate(task_dicts):
                task_id = task_dict.get("task_id", f"TASK-{i+1:03d}")
                test_tasks.append(TestTask(
                    task_id=task_id,
                    description=task_dict.get("description", ""),
                    initial_state=task_dict.get("initial_state", ""),
                    expected_outcome=task_dict.get("expected_outcome", ""),
                    priority=task_dict.get("priority", "medium"),
                    potential_bugs=task_dict.get("potential_bugs", [])
                ))
            
            self.logger.info(f"Generated {len(test_tasks)} test tasks")
            return test_tasks
            
        except Exception as e:
            self.logger.error(f"Failed to generate test tasks: {e}")
            return []

class AdaptivePlanner:
    """
    Planner that generates actions adaptively based on observed game state
    """
    def __init__(self, llm_client: OpenAI):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__ + ".AdaptivePlanner")
        self.action_history = []
        self.exploration_phase = True
        self.exploration_budget = 5  # Number of initial exploration moves
        
    def reset(self):
        """Reset the planner state for a new test"""
        self.action_history = []
        self.exploration_phase = True
        
    def plan_next_action(
        self, 
        game_description: str, 
        feature_description: str, 
        test_task: TestTask,
        current_observation: str,
        execution_results: List[ExecutionResult] = None
    ) -> ActionStep:
        """Plan the next action based on current game state and history"""
        
        # Check if we should transition from exploration to exploitation
        if self.exploration_phase and len(self.action_history) >= self.exploration_budget:
            self.exploration_phase = False
            self.logger.info("Switching from exploration to exploitation phase")
        
        # Build action history summary
        action_history_summary = self._build_action_history_summary()
        
        # Generate prompt based on current phase
        if self.exploration_phase:
            prompt = self._generate_exploration_prompt(
                game_description,
                feature_description,
                test_task,
                current_observation,
                action_history_summary
            )
        else:
            prompt = self._generate_exploitation_prompt(
                game_description,
                feature_description,
                test_task,
                current_observation,
                action_history_summary
            )
        
        # Generate next action using LLM
        try:
            response = self.llm_client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are an adaptive game testing AI that plans one action at a time"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Create an ActionStep object
            step_id = f"{'EXPLORE' if self.exploration_phase else 'TEST'}-{len(self.action_history) + 1:03d}"
            action_step = ActionStep(
                step_id=step_id,
                action_description=result.get("action_description", ""),
                expected_observation=result.get("expected_observation", ""),
                success_criteria=result.get("success_criteria", ""),
                fallback_action=result.get("fallback_action"),
                is_checkpoint=result.get("is_checkpoint", False)
            )
            
            # Store the action in history with its JSON representation
            self.action_history.append({
                "step": action_step,
                "action": result.get("action", {}),
                "phase": "exploration" if self.exploration_phase else "exploitation"
            })
            
            self.logger.info(f"Generated next action: {action_step.action_description}")
            return action_step
            
        except Exception as e:
            self.logger.error(f"Failed to generate next action: {e}")
            
            # Fallback to a basic action if generation fails
            return self._generate_fallback_action()
    
    def _generate_fallback_action(self) -> ActionStep:
        """Generate a fallback action if LLM generation fails"""
        step_id = f"{'EXPLORE' if self.exploration_phase else 'TEST'}-{len(self.action_history) + 1:03d}"
        
        # Try different directions in sequence
        directions = [
            {"type": "swipe", "start": {"x": 0.3, "y": 0.5}, "end": {"x": 0.7, "y": 0.5}},  # Right
            {"type": "swipe", "start": {"x": 0.7, "y": 0.5}, "end": {"x": 0.3, "y": 0.5}},  # Left
            {"type": "swipe", "start": {"x": 0.5, "y": 0.7}, "end": {"x": 0.5, "y": 0.3}},  # Up
            {"type": "swipe", "start": {"x": 0.5, "y": 0.3}, "end": {"x": 0.5, "y": 0.7}}   # Down
        ]
        
        direction_idx = len(self.action_history) % len(directions)
        direction_name = ["right", "left", "up", "down"][direction_idx]
        
        fallback_action = ActionStep(
            step_id=step_id,
            action_description=f"Swipe {direction_name}",
            expected_observation="Blocks may move if not obstructed",
            success_criteria="Action executed successfully",
            fallback_action=None,
            is_checkpoint=False
        )
        
        # Add to history
        self.action_history.append({
            "step": fallback_action,
            "action": directions[direction_idx],
            "phase": "exploration" if self.exploration_phase else "exploitation"
        })
        
        return fallback_action
    
    def _build_action_history_summary(self) -> str:
        """Build a summary of actions taken so far and their results"""
        if not self.action_history:
            return "No actions have been taken yet."
        
        summary = "Actions taken so far:\n"
        for i, entry in enumerate(self.action_history):
            action_step = entry["step"]
            result = entry.get("result", {})
            
            summary += f"{i+1}. {action_step.action_description}"
            
            if "success" in result:
                outcome = "succeeded" if result["success"] else "failed"
                summary += f" - {outcome}"
                
                if "observation" in result:
                    summary += f": {result['observation']}"
            
            summary += "\n"
        
        return summary
    
    def _generate_exploration_prompt(
        self, 
        game_description: str, 
        feature_description: str, 
        test_task: TestTask,
        current_observation: str,
        action_history_summary: str
    ) -> str:
        """Generate a prompt for the exploration phase"""
        
        return f"""
        You are an expert game tester exploring a puzzle game to understand its mechanics.
        
        Game Context:
        {game_description}
        
        Feature Being Tested:
        {feature_description}
        
        Test Task:
        {json.dumps(test_task.to_dict(), indent=2)}
        
        Current Game State:
        {current_observation}
        
        Action History:
        {action_history_summary}
        
        You are in the EXPLORATION PHASE. Your goal is to understand the game mechanics by trying different actions.
        Plan a SINGLE next action that helps you learn about how the game works.
        
        Provide your response as a JSON object with these fields:
        - action: A JSON object specifying exactly what to do using this format:
          {{
            "type": "swipe",
            "start": {{"x": 0.7, "y": 0.5}},  // Starting point (normalized coordinates 0-1)
            "end": {{"x": 0.3, "y": 0.5}},    // Ending point (normalized coordinates 0-1)
            "duration": 0.3
          }}
        - action_description: A clear description of what this action does and why you chose it
        - expected_observation: What you expect to happen when this action is performed
        - success_criteria: How to determine if this action revealed useful information
        - fallback_action: An alternative action if this one fails
        - is_checkpoint: false (exploration actions are never checkpoints)
        
        IMPORTANT: Use these coordinates for different swipe directions:
        - Left swipe: start {{"x": 0.7, "y": 0.5}}, end {{"x": 0.3, "y": 0.5}}
        - Right swipe: start {{"x": 0.3, "y": 0.5}}, end {{"x": 0.7, "y": 0.5}}
        - Up swipe: start {{"x": 0.5, "y": 0.7}}, end {{"x": 0.5, "y": 0.3}}
        - Down swipe: start {{"x": 0.5, "y": 0.3}}, end {{"x": 0.5, "y": 0.7}}
        """
    
    def _generate_exploitation_prompt(
        self, 
        game_description: str, 
        feature_description: str, 
        test_task: TestTask,
        current_observation: str,
        action_history_summary: str
    ) -> str:
        """Generate a prompt for the exploitation phase"""
        
        return f"""
        You are an expert game tester trying to solve a puzzle game and identify any bugs.
        
        Game Context:
        {game_description}
        
        Feature Being Tested:
        {feature_description}
        
        Test Task:
        {json.dumps(test_task.to_dict(), indent=2)}
        
        Current Game State:
        {current_observation}
        
        Action History:
        {action_history_summary}
        
        You are in the EXPLOITATION PHASE. Based on what you've learned about the game mechanics,
        your goal is now to strategically test if the level is solvable or has bugs.
        
        Plan a SINGLE next action that:
        1. Makes progress toward combining blocks into a single one
        2. Tests if certain block arrangements create unsolvable situations
        3. Verifies whether all blocks can eventually be combined
        
        Provide your response as a JSON object with these fields:
        - action: A JSON object specifying exactly what to do using this format:
          {{
            "type": "swipe",
            "start": {{"x": 0.7, "y": 0.5}},  // Starting point (normalized coordinates 0-1)
            "end": {{"x": 0.3, "y": 0.5}},    // Ending point (normalized coordinates 0-1)
            "duration": 0.3
          }}
        - action_description: A clear description of what this action does and your strategy
        - expected_observation: What you expect to happen when this action is performed
        - success_criteria: How to determine if this action revealed useful information
        - fallback_action: An alternative action if this one fails
        - is_checkpoint: Whether this step's failure would indicate a bug (use sparingly)
        
        IMPORTANT: Use these coordinates for different swipe directions:
        - Left swipe: start {{"x": 0.7, "y": 0.5}}, end {{"x": 0.3, "y": 0.5}}
        - Right swipe: start {{"x": 0.3, "y": 0.5}}, end {{"x": 0.7, "y": 0.5}}
        - Up swipe: start {{"x": 0.5, "y": 0.7}}, end {{"x": 0.5, "y": 0.3}}
        - Down swipe: start {{"x": 0.5, "y": 0.3}}, end {{"x": 0.5, "y": 0.7}}
        """
    
    def update_action_result(self, step_id: str, result: Dict[str, Any]):
        """Update the history with the result of an action"""
        for entry in self.action_history:
            if entry["step"].step_id == step_id:
                entry["result"] = result
                break
        else:
            # More robust handling - create a placeholder entry instead of raising an error
            self.logger.warning(f"Step {step_id} not found in action history, creating placeholder")
            
            # Extract action description from the result
            action_description = ""
            if isinstance(result, dict):
                action_description = result.get("action_description", "Unknown action")
            
            # Create placeholder step
            placeholder_step = ActionStep(
                step_id=step_id,
                action_description=action_description,
                expected_observation="",
                success_criteria=""
            )
            
            # Add placeholder to history
            self.action_history.append({
                "step": placeholder_step,
                "result": result,
                "phase": "unknown"
            })

class HighLevelPlanner:
    """
    Planner that generates detailed action plans for test tasks
    """
    def __init__(self, llm_client: OpenAI):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__ + ".HighLevelPlanner")
    
    def generate_action_plan(self, game_description: str, feature_description: str, test_task: TestTask):
        """Generate an action plan with mechanics exploration phase first"""
        logger.info(f"Planner agent: Creating plan for task {test_task.task_id}")
        
        prompt = f"""
        You are an expert game tester who understands how to methodically explore game mechanics before testing for bugs.
        Create a detailed testing plan that first explores how the game works, then tests specific features for bugs.
        
        Game Context:
        {game_description}
        
        Feature to Test:
        {feature_description}
        
        Test Task:
        {json.dumps(test_task.to_dict(), indent=2)}
        
        Your plan should have TWO distinct phases:
        
        PHASE 1: MECHANICS EXPLORATION (first 30-40% of steps)
        - Begin with basic observation steps to understand the current game state
        - Try each primary input/action in isolation to observe its effects
        - Experiment with combinations of successful actions
        - Document consistent patterns of what works and what doesn't
        - Form hypotheses about the game's rules and constraints
        
        PHASE 2: TARGETED TESTING (remaining 60-70% of steps)
        - Based on the mechanics learned in Phase 1, design specific tests for the feature
        - Include edge cases that might reveal bugs while respecting valid game constraints
        - Test boundary conditions where mechanics might break down
        - Verify that the game's stated objectives can be achieved
        - Check for inconsistencies in rule application
        
        IMPORTANT: Mark steps as non-checkpoint (is_checkpoint: false) if they're exploratory and their failure might represent valid game constraints.
        Only mark steps as checkpoints (is_checkpoint: true) if their failure would definitively indicate a bug.
        
        For each step, provide:
        - step_id: A unique identifier (e.g., "EXPLORE-001" or "TEST-001")
        - action_description: Clear, specific instruction to execute
        - expected_observation: What you expect to happen, including possible constraint-based failures
        - success_criteria: How to determine if the step revealed useful information (not just whether the action succeeded)
        - fallback_action: Alternative approach if the primary action doesn't produce useful results
        - is_checkpoint: Whether this step's failure would definitively indicate a bug (use sparingly)
        
        Create a sequence of steps that tests whether this level is solvable. Each step must include an 
        ACTION property that is a JSON object with the following format:
        
        For swipes:
        {{
        "type": "swipe",
        "start": {{"x": 0.5, "y": 0.7}},  // Normalized coordinates (0-1)
        "end": {{"x": 0.5, "y": 0.3}},
        "duration": 0.3
        }}
        
        For taps:
        {{
        "type": "touch",
        "action": "tap",
        "position": {{"x": 0.5, "y": 0.5}}  // Normalized coordinates (0-1)
        }}
        
        For each step, provide:
        - step_id: A unique identifier
        - description: Human-readable explanation of this step
        - action: JSON object in one of the formats above
        - expected_result: What should happen when this action is performed
        - is_checkpoint: Whether this step's failure indicates a bug
        
        Format your response as a JSON object with an array of step objects.
        """

        
        try:
            response = self.llm_client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are a specialized AI for planning game testing actions"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            step_dicts = result.get("steps", [])
            
            # Convert dictionaries to ActionStep objects
            action_steps = []
            for i, step_dict in enumerate(step_dicts):
                step_id = step_dict.get("step_id", f"STEP-{test_task.task_id}-{i+1:03d}")
                action_steps.append(ActionStep(
                    step_id=step_id,
                    action_description=step_dict.get("action_description", ""),
                    expected_observation=step_dict.get("expected_observation", ""),
                    success_criteria=step_dict.get("success_criteria", ""),
                    fallback_action=step_dict.get("fallback_action"),
                    is_checkpoint=step_dict.get("is_checkpoint", False)
                ))
            
            self.logger.info(f"Generated action plan with {len(action_steps)} steps")
            return action_steps
            
        except Exception as e:
            self.logger.error(f"Failed to generate action plan: {e}")
            return []


class InterpreterAgent:
    """
    Agent that analyzes execution results to identify bugs
    """
    def __init__(self, llm_client: OpenAI):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__ + ".InterpreterAgent")
    
    def analyze_results(
        self, 
        game_description: str, 
        feature_description: str,
        test_task: TestTask,
        execution_results: List[ExecutionResult]
    ) -> List[Bug]:
        """Analyze execution results with enhanced focus on block isolation patterns"""
        logger.info(f"Interpreter agent: Analyzing results for task {test_task.task_id}")
        
        # Convert execution_results to dictionaries for JSON serialization
        execution_results_dicts = [result.to_dict() for result in execution_results]
        
        # Extract observations to create a game state timeline
        observations = [result.observation for result in execution_results]
        
        # Count the number of actions in each direction
        action_counts = {
            "left": 0, "right": 0, "up": 0, "down": 0
        }
        
        for result in execution_results:
            action_desc = result.action_description.lower()
            if "left" in action_desc:
                action_counts["left"] += 1
            elif "right" in action_desc:
                action_counts["right"] += 1
            elif "up" in action_desc:
                action_counts["up"] += 1
            elif "down" in action_desc:
                action_counts["down"] += 1
        
        prompt = f"""
        You are an expert analyst specializing in puzzle game mechanics. Your task is to determine if the current level
        of a jelly block sliding puzzle game is solvable or contains a bug making it impossible to win.
        
        Game Context:
        {game_description}
        
        Feature Being Tested:
        {feature_description}
        
        Test Task:
        {json.dumps(test_task.to_dict(), indent=2)}
        
        Testing Details:
        - We've performed {len(execution_results)} actions on this level
        - Actions by direction: {json.dumps(action_counts)}
        
        PLEASE ANALYZE THE GAME STATE OBSERVATIONS CAREFULLY for evidence of blocks being permanently isolated:
        
        Timeline of observations:
        {json.dumps(observations, indent=2)}
        
        Execution Results:
        {json.dumps(execution_results_dicts, indent=2)}
        
        CRUCIAL ANALYSIS QUESTIONS:
        
        1. ISOLATION ANALYSIS:
        - Are there blocks that appear to be permanently separated by walls or obstacles?
        - Are there blocks in separate "chambers" with no connecting path between them?
        - Does the grid structure prevent certain blocks from ever meeting?
        
        2. MOVEMENT PATTERN ANALYSIS:
        - Have we tried all four directions (up, down, left, right) multiple times?
        - Do blocks stop moving in certain directions due to walls/obstacles?
        - Are there blocks that never change position despite multiple swipe attempts?
        
        3. MATHEMATICAL POSSIBILITY:
        - Based on the grid layout and block positions, is it theoretically possible to combine all blocks?
        - Are there subgroups of blocks that can combine within themselves but cannot merge with other subgroups?
        - Is there a path that would allow all blocks to eventually meet?
        
        4. WIN CONDITION ASSESSMENT:
        - Has the level already been completed (all blocks merged into one)?
        - If not, is there a clear strategy to combine all remaining blocks?
        - Is there mathematical proof that the remaining configuration cannot be solved?
        
        IMPORTANT: A level is UNSOLVABLE if and only if there is a MATHEMATICAL CERTAINTY that blocks cannot all be combined,
        not just because our specific sequence of moves didn't solve it.
        
        For each bug you identify, provide:
        - bug_id: A unique identifier
        - description: Detailed explanation of why the level is unsolvable with specific reference to block positions
        - severity: How seriously it impacts gameplay ("critical" for unsolvable levels)
        - reproduction_steps: Which level configuration demonstrates this isolation bug
        - expected_behavior: All blocks should be combinable into one
        - actual_behavior: Detailed explanation of the isolation pattern making combining impossible
        - potential_fix: Suggested modifications to make the level solvable
        
        If the level is solvable (no bugs), explain why and provide:
        - A clear explanation of why blocks can be combined
        - Which moves would lead to a solution
        - If the level has already been solved
        
        Format your response as a JSON object with:
        1. An array of bug objects under the "bugs" key (empty if no bugs)
        2. A "solvability_analysis" field explaining your reasoning about solvability
        3. A "win_status" field indicating if the level is "solved", "solvable", or "unsolvable"
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are a specialized AI for identifying game bugs and analyzing puzzle solvability"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            bug_dicts = result.get("bugs", [])
            
            # Log the solvability analysis
            solvability_analysis = result.get("solvability_analysis", "No analysis provided")
            win_status = result.get("win_status", "unknown")
            self.logger.info(f"Solvability analysis: {win_status}")
            self.logger.info(f"Analysis details: {solvability_analysis[:100]}...")
            
            # Convert dictionaries to Bug objects
            bugs = []
            for i, bug_dict in enumerate(bug_dicts):
                bug_id = bug_dict.get("bug_id", f"BUG-{test_task.task_id}-{i+1:03d}")
                bugs.append(Bug(
                    bug_id=bug_id,
                    description=bug_dict.get("description", ""),
                    severity=bug_dict.get("severity", "critical"),
                    reproduction_steps=bug_dict.get("reproduction_steps", ""),
                    expected_behavior=bug_dict.get("expected_behavior", ""),
                    actual_behavior=bug_dict.get("actual_behavior", ""),
                    affected_tasks=bug_dict.get("affected_tasks", [test_task.task_id]),
                    potential_fix=bug_dict.get("potential_fix")
                ))
            
            self.logger.info(f"Identified {len(bugs)} bugs")
            return bugs
            
        except Exception as e:
            self.logger.error(f"Failed to analyze results: {e}")
            return []



class GameTestingPipeline:
    """
    Main pipeline for testing game features
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize game testing pipeline"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + ".GameTestingPipeline")
        
        # Set up OpenAI client
        openai_api_key = self.config.get("openai_api_key", os.environ.get("OPENAI_API_KEY"))
        if not openai_api_key:
            raise ValueError("OpenAI API key is required")
        self.llm_client = OpenAI(api_key=openai_api_key)
        
        # Initialize agents
        self.tester_agent = TesterAgent(self.llm_client)
        self.planner = HighLevelPlanner(self.llm_client)
        self.interpreter = InterpreterAgent(self.llm_client)
        
        # Initialize SIMA - ensure SIMAAgent accepts config
        self.sima_agent = SIMAAgent(self.config)
        
        # Initialize video processing components
        self.logger.info("Initializing video processing components")
        self.video_observer = UnityVideoObserver(self.config)
        self.video_processor = self.video_observer.get_processor()
        
        # Thread pool for parallel task execution
        self.max_workers = self.config.get("max_workers", 3)
        
        self.logger.info("Game Testing Pipeline initialized with video processing capabilities")

    
    def run_pipeline(
        self, 
        game_description: str, 
        feature_description: str,
        num_tasks: int = 10
    ) -> List[Bug]:
        """Run the full testing pipeline"""
        self.logger.info("Starting testing pipeline")
        
        # Step 1: Generate test tasks
        test_tasks = self.tester_agent.generate_test_tasks(
            game_description, 
            feature_description,
            num_tasks
        )
        
        if not test_tasks:
            self.logger.warning("No test tasks generated")
            return []
        
        self.logger.info(f"Generated {len(test_tasks)} test tasks")
        
        # Step 2-5: Execute test tasks in parallel and collect bugs
        all_bugs = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all test tasks to the executor
            future_to_task = {
                executor.submit(
                    self._execute_test_task, 
                    game_description, 
                    feature_description, 
                    task
                ): task for task in test_tasks
            }
            
            # Process results as they complete
            for future in future_to_task:
                task = future_to_task[future]
                try:
                    task_bugs = future.result()
                    if task_bugs:
                        all_bugs.extend(task_bugs)
                        self.logger.info(f"Task {task.task_id} found {len(task_bugs)} bugs")
                    else:
                        self.logger.info(f"Task {task.task_id} completed successfully with no bugs")
                except Exception as e:
                    self.logger.error(f"Error executing task {task.task_id}: {str(e)}")
        
        self.logger.info(f"Testing pipeline completed. Found {len(all_bugs)} bugs in total.")
        return all_bugs
    
    def _execute_test_task(
        self, 
        game_description: str, 
        feature_description: str, 
        test_task: TestTask
    ) -> List[Bug]:
        """Execute a single test task"""
        self.logger.info(f"Executing test task {test_task.task_id}: {test_task.description}")
        
        # Generate action plan
        action_plan = self.planner.generate_action_plan(
            game_description,
            feature_description,
            test_task
        )
        
        if not action_plan:
            self.logger.warning(f"No action plan generated for task {test_task.task_id}")
            return []
        
        # Execute action plan
        execution_results = self._execute_action_plan(action_plan)
        
        # Analyze results
        bugs = self.interpreter.analyze_results(
            game_description,
            feature_description,
            test_task,
            execution_results
        )
        
        return bugs
    
    def _execute_test_task_adaptive(
        self, 
        game_description: str, 
        feature_description: str, 
        test_task: TestTask
    ):
        """Execute a test task using the adaptive planner with video processing and feedback loop"""
        self.logger.info(f"Executing test task with adaptive planning and video analysis: {test_task.task_id}")
        
        # Initialize the adaptive planner
        planner = AdaptivePlanner(self.llm_client)
        
        # Start capturing video frames
        self.logger.info("Starting video frame capture")
        for _ in range(5):  # Capture initial frames
            self.video_observer.update()
            time.sleep(0.1)
        
        # Get initial observation using traditional method and video processing
        current_observation = self._get_current_observation()
        
        # Analyze initial game situation using video
        initial_situation = self.video_processor.analyze_game_situation(
            f"Test task: {test_task.description}\nExpected outcome: {test_task.expected_outcome}"
        )
        self.logger.info(f"Initial situation analysis: {initial_situation.get('analysis', 'No analysis available')}")
        
        execution_results = []
        max_steps = 15  # Prevent infinite loops
        step_count = 0
        
        # Execute actions until task complete or max steps reached
        while not self._is_task_complete(execution_results, test_task) and step_count < max_steps:
            step_count += 1
            self.logger.info(f"Planning step {step_count}/{max_steps}")
            
            # Determine the next action to take using video-based analysis
            action_recommendation = self.video_processor.determine_next_action(
                game_description,
                test_task.to_dict()
            )
            
            # Extract action from recommendation
            if isinstance(action_recommendation, dict) and "description" in action_recommendation:
                action_text = action_recommendation["description"]
                self.logger.info(f"Video processor recommended action: {action_text}")
                
                # Create an ActionStep from the recommended action
                step_id = f"{'EXPLORE' if step_count <= 5 else 'TEST'}-{step_count:03d}"
                action = ActionStep(
                    step_id=step_id,
                    action_description=action_text,
                    expected_observation="Expected changes in game state",
                    success_criteria="Action executed successfully",
                    fallback_action=None,
                    is_checkpoint=False
                )
                
                # Add to planner's action history
                planner.action_history.append({
                    "step": action,
                    "action": {"type": "custom", "description": action_text},
                    "phase": "exploration" if step_count <= 5 else "exploitation"
                })
            else:
                # Fall back to adaptive planner if video processor couldn't determine action
                self.logger.info("Falling back to traditional adaptive planner")
                action = planner.plan_next_action(
                    game_description,
                    feature_description,
                    test_task,
                    current_observation,
                    execution_results
                )
            
            if not action:
                self.logger.warning("No action planned, ending task execution")
                break
                
            # Execute the action
            action_text = action.action_description if isinstance(action, ActionStep) else action
            self.logger.info(f"Executing action: {action_text}")
            result = self.sima_agent.execute_action(action_text)
            
            # Adaptive wait time based on action complexity
            wait_time = 0.5  # Default wait time
            
            # If action seems complex, wait longer
            action_complexity_indicators = ['multiple', 'sequence', 'complex', 'carefully']
            if any(indicator in action_text.lower() for indicator in action_complexity_indicators):
                wait_time = 1.0  # Longer wait for complex actions
                
            time.sleep(wait_time)
            
            # Update video observation to capture the result - more efficient approach
            max_frames = 3
            frames_captured = 0
            last_change_detected = False
            
            # Capture frames until we either see a change or hit our limit
            while frames_captured < max_frames:
                self.video_observer.update()
                frames_captured += 1
                
                # Only wait between frames if we haven't captured enough yet
                if frames_captured < max_frames:
                    time.sleep(0.05)  # Shorter wait between frame captures
            
            # Get updated observation using traditional method
            current_observation = self._get_current_observation()
            
            # Create execution result
            # Use the actual step ID from the action object returned by the planner
            if isinstance(action, ActionStep):
                # Use the step ID directly from the ActionStep object
                step_id = action.step_id
                action_description = action.action_description
            else:
                # Fallback if action is a string (backward compatibility)
                step_id = f"{'EXPLORE' if step_count <= 5 else 'TEST'}-{step_count:03d}"
                action_description = action
                
            execution_result = ExecutionResult(
                step_id=step_id,
                action_description=action_description,
                observation=current_observation,
                success=result.get("success", False),
                raw_data=result
            )
            
            # Add to results
            execution_results.append(execution_result)
            
            # Update planner with the result
            planner.update_action_result(execution_result.step_id, execution_result.to_dict())
            
        # Analyze the results with video context
        self.logger.info("Analyzing execution results with video context")
        
        # First, get final video-based situation analysis
        final_situation = self.video_processor.analyze_game_situation(
            f"Test task: {test_task.description}\nExpected outcome: {test_task.expected_outcome}"
        )
        self.logger.info(f"Final situation analysis: {final_situation.get('analysis', 'No analysis available')}")
        
        # Add video analysis to the last execution result
        if execution_results and 'analysis' in final_situation:
            execution_results[-1].raw_data['video_analysis'] = final_situation['analysis']
        
        # Analyze with interpreter
        bugs = self.interpreter.analyze_results(
            game_description,
            feature_description,
            test_task,
            execution_results
        )
        
        return bugs
        
    def _is_task_complete(self, execution_results, test_task):
        """Determine if a test task has been completed based on execution results"""
        # If we don't have enough results, task isn't complete
        if not execution_results or len(execution_results) < 2:
            return False
            
        # Get the last execution result
        last_result = execution_results[-1]
        
        # Check if we have any success indicator in the raw data
        if last_result.raw_data.get('task_complete') is True:
            return True
            
        # Check if the last observation contains indicators of completion
        expected_outcome_terms = test_task.expected_outcome.lower().split()
        observation = last_result.observation.lower()
        
        # Count how many expected outcome terms appear in the observation
        matches = sum(1 for term in expected_outcome_terms if term in observation and len(term) > 3)
        match_threshold = len(expected_outcome_terms) // 2  # At least half the terms should match
        
        # If we have success flag and enough matching terms, task is complete
        if last_result.success and matches >= match_threshold:
            return True
            
        # If we have video analysis in the raw data, check that too
        if 'video_analysis' in last_result.raw_data:
            video_analysis = last_result.raw_data['video_analysis'].lower()
            video_matches = sum(1 for term in expected_outcome_terms if term in video_analysis and len(term) > 3)
            if video_matches >= match_threshold:
                return True
                
        # If we've reached max steps, consider the task complete for practical purposes
        if len(execution_results) >= 12:  # Using 12 as a practical limit (less than the max_steps=15)
            return True
            
        return False

    # Simple cache for frame descriptions to avoid redundant API calls
    _frame_description_cache = {}
    _max_cache_size = 10
    
    def _get_current_observation(self):
        """Get a detailed description of the current game state using both screenshot and video analysis"""
        try:
            # Check if we have video frames available
            if hasattr(self, 'video_processor') and self.video_processor and len(self.video_processor.frame_buffer) > 0:
                self.logger.info("Using video processor for game state observation")
                
                # Get the most recent frame
                recent_frame = self.video_processor.frame_buffer[-1]["frame"]
                
                # Resize the frame to reduce data size (800x600 is plenty for analysis)
                max_size = (800, 600)
                if recent_frame.width > max_size[0] or recent_frame.height > max_size[1]:
                    recent_frame = recent_frame.resize(max_size, Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
                
                # Convert RGBA to RGB mode before saving as JPEG
                if recent_frame.mode == 'RGBA':
                    recent_frame = recent_frame.convert('RGB')
                    
                # Convert the frame to base64 with optimized compression
                buffer = io.BytesIO()
                recent_frame.save(buffer, format="JPEG", quality=85, optimize=True)
                screenshot = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                # Check if we have a cached result for a similar image
                # Use the first 100 chars of the base64 string as a simple hash
                image_hash = screenshot[:100]
                if image_hash in self._frame_description_cache:
                    self.logger.info("Using cached observation for similar frame")
                    return self._frame_description_cache[image_hash]
                
                # Construct a prompt appropriate for general 3D games
                prompt = """
                Describe this 3D game state in detail. Focus on:
                1. The player character position and state
                2. The environment and visible objects
                3. Any NPCs, enemies, or interactive elements
                4. The apparent objective or current mission
                5. Any physics interactions or dynamic elements
                6. Any unusual visual artifacts or glitches
                7. Spatial relationships and navigation possibilities
                """
            else:
                # Fall back to traditional screenshot method
                self.logger.info("Falling back to traditional screenshot method")
                screenshot = self.sima_agent.get_screenshot()
                
                # If we don't have a screenshot, return a default observation
                if screenshot is None:
                    return "The game state could not be observed. No screenshot available."
                
                # Construct a prompt appropriate for jelly block game
                prompt = """
                Describe this jelly block puzzle game state in detail. Focus on:
                1. The arrangement of blocks on the grid
                2. The colors, sizes, and numbers on blocks
                3. Any special blocks or power-ups
                4. The overall objective based on what you see
                5. The current progress toward that objective
                6. Any blocks that appear to be isolated or unable to merge
                """
            
            # Use OpenAI's vision model to describe the game state
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a specialized vision analyst for video games. You provide detailed observations about the current game state to assist with testing and bug detection."
                    },
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text", 
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            description = response.choices[0].message.content
            
            # Cache the result for similar frames
            if screenshot:
                image_hash = screenshot[:100]
                self._frame_description_cache[image_hash] = description
                
                # Limit cache size by removing oldest entries
                if len(self._frame_description_cache) > self._max_cache_size:
                    oldest_key = next(iter(self._frame_description_cache))
                    self._frame_description_cache.pop(oldest_key)
            
            return description
            
        except Exception as e:
            self.logger.error(f"Error getting observation: {str(e)}")
            return "Failed to get observation due to an error."

def test_current_level():
    """Test the currently visible level using video processing and adaptive planning"""
    print("\n===== Testing Current Level With Video Processing =====\n")
    
    # Game and feature descriptions for a 3D game
    game_description = """3D Adventure Game is an open-world exploration game where players navigate through various environments,
    solve puzzles, and interact with objects and characters.
    
    The game features:
    - A 3D environment with physics simulation
    - Player character with movement abilities (walk, run, jump, interact)
    - Various interactive objects that can be manipulated
    - Environmental puzzles that require spatial reasoning
    - Character progression and inventory system
    """
    
    feature_description = """The physics interaction system allows players to manipulate objects in the game world
    with realistic physical properties. Objects should respond to gravity, collisions, and player interactions
    in a consistent and predictable manner.
    
    The feature implementation ensures that:
    1. Objects have appropriate mass, friction, and collision properties
    2. Objects can be pushed, pulled, picked up, and thrown based on their size and weight
    3. Physics interactions do not cause objects to clip through walls or other solid objects
    4. Objects at rest remain stable and do not jitter or move unexpectedly
    5. Collisions between objects produce appropriate responses and sound effects
    """
    
    # Create a test task for a 3D game
    test_task = TestTask(
        task_id="physics_interaction_test",
        description="Test if physics interactions in the 3D environment work correctly and consistently",
        initial_state="Player is in a 3D environment with various interactive objects visible",
        expected_outcome="Objects should respond realistically to player interactions, respect collision boundaries, and not exhibit glitches or unexpected behavior"
    )
    
    # Create pipeline instance with configuration for video processing
    config = {
        "video_buffer_size": 8,  # Store 8 frames for analysis
        "capture_fps": 15,      # Capture at 15 frames per second
        "resize_shape": (800, 600),  # Resolution for captures
        "change_threshold": 0.1  # Sensitivity for detecting changes
    }
    
    pipeline = GameTestingPipeline(config)
    
    print("Starting test with video processing enabled")
    print("This will analyze the game through real-time video frames")
    print("Please ensure the game window is visible and active\n")
    
    # Give user time to prepare
    for i in range(5, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    # Execute the test with video processing
    bugs = pipeline._execute_test_task_adaptive(
        game_description,
        feature_description,
        test_task
    )
    
    # Display results
    if bugs:
        print(f"\n{len(bugs)} bug(s) found:")
        for bug in bugs:
            print(f"\nBug ID: {bug.bug_id}")
            print(f"Description: {bug.description}")
            print(f"Severity: {bug.severity}")
            print(f"Steps to reproduce: {bug.reproduction_steps}")
    else:
        print("\nNo bugs found.")

test_current_level()
