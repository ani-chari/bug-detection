import os
import json
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import required libraries
import torch
import numpy as np
from PIL import Image
from openai import OpenAI

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
            raise ValueError(f"Step {step_id} not found in action history")

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
        # self.sima_agent = SIMAAgent(self.config)
        self.sima_agent = SIMACore()
        
        # Thread pool for parallel task execution
        self.max_workers = self.config.get("max_workers", 3)
        
        self.logger.info("Game Testing Pipeline initialized")

    
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
    ) -> List[Bug]:
        """Execute a test task using the adaptive planner with feedback loop"""
        self.logger.info(f"Executing test task {test_task.task_id} adaptively: {test_task.description}")
        
        # Initialize the adaptive planner
        adaptive_planner = AdaptivePlanner(self.llm_client)
        adaptive_planner.reset()
        
        # Initialize execution results
        execution_results = []
        
        # Maximum number of steps to prevent infinite loops
        max_steps = 20
        
        # Execute steps adaptively
        for step_num in range(max_steps):
            # Get current observation (screenshot + description)
            current_observation = self._get_current_observation()
            
            # Plan next action based on current state
            action_step = adaptive_planner.plan_next_action(
                game_description,
                feature_description,
                test_task,
                current_observation,
                execution_results
            )
            
            # Extract the action JSON from the planner's history
            action_json = None
            for entry in adaptive_planner.action_history:
                if entry["step"].step_id == action_step.step_id:
                    action_json = entry.get("action", {})
                    break
            
            # Execute the action using the controller directly with JSON
            self.logger.info(f"Executing step {action_step.step_id}: {action_step.action_description}")
            
            # Either execute via SIMA or directly via controller
            if action_json and self.sima_agent and hasattr(self.sima_agent, "controller"):
                # Execute directly with controller
                controller_result = self.sima_agent.controller.execute([action_json])
                result = {
                    "success": controller_result.get("success", False),
                    "observation": f"Executed {action_json.get('type', 'unknown')} action",
                    "raw_data": controller_result
                }
            else:
                # Fall back to traditional execution
                result = self.sima_agent.execute_action(action_step.action_description)
            
            # Get updated observation after action
            post_action_observation = self._get_current_observation()
            
            # Create execution result
            execution_result = ExecutionResult(
                step_id=action_step.step_id,
                action_description=action_step.action_description,
                observation=post_action_observation,
                success=result.get("success", False),
                raw_data=result
            )
            
            # Update planner with the result
            adaptive_planner.update_action_result(action_step.step_id, {
                "success": execution_result.success,
                "observation": execution_result.observation
            })
            
            # Add to execution results
            execution_results.append(execution_result)
            
            # Handle failures
            if not execution_result.success:
                self.logger.warning(f"Step {action_step.step_id} failed")
                
                # Checkpoint failures abort the plan
                if action_step.is_checkpoint:
                    self.logger.warning(f"Checkpoint step {action_step.step_id} failed. Aborting plan.")
                    break
                
                # Try fallback action if available
                if action_step.fallback_action:
                    # Execute fallback logic
                    # (Similar to above execution logic)
                    pass
            
            # Check if we've completed the task (combined all blocks or determined impossible)
            if self._is_task_complete(execution_results, test_task):
                self.logger.info(f"Task {test_task.task_id} completed successfully")
                break
        
        # Analyze results to find bugs
        bugs = self.interpreter.analyze_results(
            game_description,
            feature_description,
            test_task,
            execution_results
        )
        
        return bugs

    def _get_current_observation(self) -> str:
        """Get a detailed description of the current game state with focus on block isolation"""
        try:
            # Use LLM to describe the current screenshot
            if hasattr(self.sima_agent, "observer") and self.sima_agent.observer:
                # Get screenshot
                screenshot = self.sima_agent.observer.get_observation()
                
                # Analyze screenshot using OpenAI Vision API
                import base64
                from io import BytesIO
                
                # Convert tensor to PIL Image
                from torchvision.transforms import ToPILImage
                transform = ToPILImage()
                pil_image = transform(screenshot)
                
                # Convert to base64 for API
                buffered = BytesIO()
                pil_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                prompt = """
                Analyze this jelly block puzzle game screenshot with extreme precision. Focus on:
                
                1. DETAILED GRID LAYOUT:
                - Exact dimensions (e.g., 5x5)
                - Position of walls and obstacles
                - Whether obstacles create separate chambers/regions
                
                2. BLOCK ANALYSIS:
                - Exact number of blocks
                - Precise position of each block (e.g., "Block 1 at position (2,3)")
                - Whether blocks are in separate chambers/regions
                
                3. ISOLATION ASSESSMENT:
                - Are any blocks permanently isolated by walls?
                - Can all blocks potentially reach each other through some sequence of moves?
                - Are there any mathematically impossible configurations?
                
                4. MOVEMENT POSSIBILITIES:
                - Which directions would cause blocks to move (up/down/left/right)
                - Which blocks would collide if moved in each direction
                - What pattern of moves might lead to combining all blocks
                
                Provide a detailed, analytical description that would help determine if this level is solvable.
                """
                
                # Call Vision API
                try:
                    response = self.llm_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                            ]}
                        ],
                        max_tokens=500
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    self.logger.error(f"Error analyzing screenshot: {e}")
                    return "Game screen shows a puzzle with jelly blocks on a grid."
            else:
                return "Current game state unknown."
        except Exception as e:
            self.logger.error(f"Error getting current observation: {e}")
            return "Unable to capture current game state."


    def _is_task_complete(self, execution_results: List[ExecutionResult], test_task: TestTask) -> bool:
        """Check if the task has been completed based on recent observations and game state patterns"""
        if not execution_results:
            return False
            
        # Check if we've already tried all four directions multiple times
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
        
        # Check if we've tried each direction at least twice
        thorough_testing = all(count >= 2 for count in action_counts.values())
        
        # Check last observation for signs of completion or impossibility
        last_observation = execution_results[-1].observation.lower()
        
        # Solved indicators
        solved_indicators = [
            "all blocks combined", 
            "single block", 
            "level completed", 
            "level solved",
            "only one block remains",
            "successfully merged all blocks"
        ]
        
        # Unsolvable indicators
        unsolvable_indicators = [
            "impossible to combine",
            "permanently isolated",
            "no path between",
            "cannot merge",
            "separate chambers",
            "blocks can never meet",
            "mathematically impossible"
        ]
        
        # Check for win condition (solved)
        if any(indicator in last_observation for indicator in solved_indicators):
            self.logger.info("Level appears to be SOLVED based on observations")
            return True
            
        # Check for unsolvable condition
        if thorough_testing and any(indicator in last_observation for indicator in unsolvable_indicators):
            self.logger.info("Level appears to be UNSOLVABLE based on observations")
            return True
        
        # Check for sequence of no changes
        if len(execution_results) >= 5:
            last_five = execution_results[-5:]
            no_change_count = 0
            
            for result in last_five:
                if any(phrase in result.observation.lower() for phrase in 
                    ["no change", "nothing happened", "blocks didn't move", "remained the same"]):
                    no_change_count += 1
            
            # If we've had 5 consecutive no-change observations after trying different directions
            if no_change_count >= 5 and thorough_testing:
                self.logger.info("Task may be complete - 5 consecutive actions with no change")
                return True
        
        # Not complete yet
        return False


    
    def _execute_action_plan(self, action_plan: List[ActionStep]) -> List[ExecutionResult]:
        """Execute an action plan using the SIMA agent"""
        self.logger.info(f"Executing action plan with {len(action_plan)} steps")
        
        execution_results = []
        
        for step in action_plan:
            self.logger.info(f"Executing step {step.step_id}: {step.action_description}")
            
            # Execute the action using SIMA
            result = self.sima_agent.execute_action(step.action_description)
            
            # Create execution result
            execution_result = ExecutionResult(
                step_id=step.step_id,
                action_description=step.action_description,
                observation=result.get("observation", ""),
                success=result.get("success", False),
                raw_data=result
            )
            
            execution_results.append(execution_result)
            
            # Check if the step failed
            if not execution_result.success:
                self.logger.warning(f"Step {step.step_id} failed")
                
                # If this is a checkpoint step, abort the plan
                if step.is_checkpoint:
                    self.logger.warning(f"Checkpoint step {step.step_id} failed. Aborting plan.")
                    break
                
                # Try fallback action if available
                if step.fallback_action:
                    self.logger.info(f"Attempting fallback action for step {step.step_id}")
                    
                    fallback_result = self.sima_agent.execute_action(step.fallback_action)
                    
                    fallback_execution_result = ExecutionResult(
                        step_id=f"{step.step_id}_fallback",
                        action_description=step.fallback_action,
                        observation=fallback_result.get("observation", ""),
                        success=fallback_result.get("success", False),
                        raw_data=fallback_result,
                        is_fallback=True
                    )
                    
                    execution_results.append(fallback_execution_result)
                    
                    # If the fallback also failed and this is a checkpoint, abort the plan
                    if not fallback_execution_result.success and step.is_checkpoint:
                        self.logger.warning(f"Fallback for checkpoint step {step.step_id} also failed. Aborting plan.")
                        break
        
        return execution_results


def main():
    """Example usage of the game testing pipeline"""
    # Configuration (customize as needed)
    config = {
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "max_workers": 2,
        "sima_config": {
            # SIMA configuration
            "vision_encoder": "openai/clip-vit-large-patch14",
            "text_encoder": "sentence-transformers/all-mpnet-base-v2",
            "env_type": "unity",
            "unity_executable_path": "/path/to/your/game.exe",  # Replace with actual path
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            
            # Custom action space for your game (optional)
            "action_space": [
                "move", "look", "click", "interact", "use_item",
                "attack", "jump", "open_menu", "close_menu"
            ],
            "observer": {
                "capture_method": "adb",
                "resize_shape": (224, 224)
            },
            "controller": {
                "control_method": "adb",
                "action_delay": 0.5
            }

        }
    }
    
    game_description = """
    'Jelly Merge Puzzle' is a grid-based sliding puzzle game where jelly blocks are arranged on a square grid. Players can swipe in four directions (up, down, left, right) to slide all blocks simultaneously in that direction. Blocks move until they hit an obstacle, the edge of the grid, or another block. When two blocks collide, they combine into a single block and continue sliding together. The goal of each level is to combine all blocks into a single block through a sequence of strategic swipes. The game progresses through multiple levels with increasingly complex grid configurations and starting block arrangements.

    Key mechanics:
    1. All blocks move simultaneously when the player swipes
    2. Blocks combine when they collide
    3. Combined blocks continue sliding in the swipe direction
    4. Blocks stop when hitting walls or grid edges
    5. The win condition requires combining all blocks into a single block
    6. Some levels have walls or blocked grid squares that affect movement
    """

    feature_description = """
    Level solvability feature: Each level in the game should be designed to have at least one valid solution where all blocks can be combined into a single block through some sequence of swipe actions. The game should not contain any levels with impossible-to-win configurations where blocks cannot all be combined regardless of the sequence of moves used.
    """

    
    # Initialize and run the pipeline
    try:
        pipeline = GameTestingPipeline(config)
        bugs = pipeline.run_pipeline(game_description, feature_description, num_tasks=5)
        
        # Output the results
        print("\n=== TESTING RESULTS ===\n")
        if bugs:
            print(f"Found {len(bugs)} bugs:")
            for i, bug in enumerate(bugs, 1):
                print(f"\nBug #{i}: {bug.bug_id}")
                print(f"Description: {bug.description}")
                print(f"Severity: {bug.severity}")
                print(f"Reproduction steps: {bug.reproduction_steps}")
                print(f"Expected behavior: {bug.expected_behavior}")
                print(f"Actual behavior: {bug.actual_behavior}")
        else:
            print("No bugs found.")
        print("\n=== END OF TESTING RESULTS ===\n")
        
    except Exception as e:
        print(f"Error running the pipeline: {str(e)}")


def test_current_level():
    """Test the currently visible level using adaptive planning"""
    device_id = None
    try:
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')[1:]
        
        for line in lines:
            if line.strip() and "localhost" in line and "device" in line:
                device_id = line.split()[0]
                print(f"Found BlueStacks device: {device_id}")
                break
    except Exception as e:
        print(f"Error finding device: {e}")
    
    config = {
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "max_workers": 1,
        "sima_config": {
            "observer": {
                "capture_method": "adb",
                "resize_shape": (224, 224),
                "device_id": device_id
            },
            "controller": {
                "control_method": "adb",
                "action_delay": 0.5,
                "device_id": device_id
            }
        }
    }
    
    # Game descriptions
    game_description = """
    'Jelly Merge Puzzle' is a grid-based sliding puzzle game where jelly blocks are arranged on a square grid. Players can swipe in four directions (up, down, left, right) to slide all blocks simultaneously in that direction. Blocks move until they hit an obstacle, the edge of the grid, or another block. When two blocks collide, they combine into a single block and continue sliding together. The goal of each level is to combine all blocks into a single block through a sequence of strategic swipes.
    """
    
    feature_description = """
    Level solvability feature: Each level should be designed to have at least one valid solution where all blocks can be combined into a single block through some sequence of swipe actions. The game should not contain any levels with impossible-to-win configurations.
    """
    
    # Initialize pipeline
    pipeline = GameTestingPipeline(config)
    
    # Create a simple test task for the current level
    test_task = TestTask(
        task_id="TEST-CURRENT-LEVEL",
        description="Test if the current level is solvable by finding a sequence of swipes that combines all blocks",
        initial_state="Game is showing a level with jelly blocks arranged on a grid",
        expected_outcome="All blocks can be combined into a single block through some sequence of swipes",
        potential_bugs=["Impossible-to-win configuration", "Isolated blocks", "Blocks that cannot be combined"]
    )
    
    # Run the adaptive test
    bugs = pipeline._execute_test_task_adaptive(game_description, feature_description, test_task)
    
    if bugs:
        print("\n==== BUGS DETECTED ====")
        for bug in bugs:
            print(f"Bug ID: {bug.bug_id}")
            print(f"Description: {bug.description}")
            print(f"Severity: {bug.severity}")
            print(f"Expected: {bug.expected_behavior}")
            print(f"Actual: {bug.actual_behavior}")
            print(f"Fix suggestion: {bug.potential_fix}\n")
    else:
        print("\n==== NO BUGS DETECTED ====")
        print("The current level appears to be solvable.")

if __name__ == "__main__":
    test_current_level()
