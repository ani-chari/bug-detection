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
        
        Format your response as a JSON object with an array of step objects under the "steps" key.
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
        """Analyze execution results to identify genuine bugs vs. valid game constraints"""
        logger.info(f"Interpreter agent: Analyzing results for task {test_task.task_id}")
        
        # Convert execution_results to dictionaries for JSON serialization
        execution_results_dicts = [result.to_dict() for result in execution_results]
        
        prompt = f"""
        You are an expert game analyst with deep experience in understanding game mechanics across many genres. 
        Your task is to analyze test results to identify genuine bugs while accounting for valid game constraints.
        
        Game Context:
        {game_description}
        
        Feature Being Tested:
        {feature_description}
        
        Test Task:
        {json.dumps(test_task.to_dict(), indent=2)}
        
        Execution Results:
        {json.dumps(execution_results_dicts, indent=2)}
        
        IMPORTANT: In games, unsuccessful actions are often part of valid game mechanics, not bugs.
        
        First, analyze the test results to understand the game's core mechanics and constraints:
        1. What actions succeed or fail consistently?
        2. Are there patterns to when actions fail?
        3. Do the failures align with the described game mechanics?
        4. What appears to be normal/expected behavior for this game type?
        
        Then, identify actual bugs by looking for:
        1. Inconsistencies where similar actions in similar contexts have different outcomes
        2. Actions that should work according to the game's rules but fail
        3. Behavior that contradicts the stated design goals
        4. Actions that render the game's objectives impossible to achieve
        5. Contradictions in the game's own established rules
        
        For each genuine bug you identify, provide:
        - bug_id: A unique identifier
        - description: Detailed explanation of the bug
        - severity: How seriously it impacts gameplay ("critical", "high", "medium", "low")
        - reproduction_steps: Actions that reliably demonstrate the issue
        - expected_behavior: What should happen according to game rules
        - actual_behavior: What actually happened that contradicts game rules
        - potential_fix: Suggested solution that respects the game's mechanics
        
        If you determine that failures are valid constraints rather than bugs, return an empty array with reasoning.
        
        Format your response as a JSON object with an array of bug objects under the "bugs" key and optional "mechanics_insights" 
        explaining your understanding of the game's constraints.
        """
        
        try:
            response = self.llm_client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {"role": "system", "content": "You are a specialized AI for identifying game bugs"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            bug_dicts = result.get("bugs", [])
            
            # Convert dictionaries to Bug objects
            bugs = []
            for i, bug_dict in enumerate(bug_dicts):
                bug_id = bug_dict.get("bug_id", f"BUG-{test_task.task_id}-{i+1:03d}")
                bugs.append(Bug(
                    bug_id=bug_id,
                    description=bug_dict.get("description", ""),
                    severity=bug_dict.get("severity", "medium"),
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


# Simple test script for current level
def test_current_level():
    # Find the connected BlueStacks device
    device_id = None
    try:
        import subprocess
        result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')[1:]
        
        # Look for BlueStacks device (localhost)
        for line in lines:
            if line.strip() and "localhost" in line and "device" in line:
                device_id = line.split()[0]
                print(f"Found BlueStacks device: {device_id}")
                break
                
        # If no BlueStacks found, use any device
        if not device_id and lines:
            device_id = lines[0].split()[0]
            print(f"Using device: {device_id}")
    except Exception as e:
        print(f"Error finding device: {e}")
    
    if not device_id:
        print("No devices found! Please make sure BlueStacks is running and ADB is connected.")
        return
    
    config = {
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "max_workers": 1,
        "sima_config": {
            "observer": {
                "capture_method": "adb",  # Explicit capture method
                "resize_shape": (224, 224),
                "device_id": device_id    # Explicitly set device ID
            },
            "controller": {
                "control_method": "adb",  # Explicit control method
                "action_delay": 0.5,
                "device_id": device_id    # Explicitly set device ID
            }
        }
    }
    
    # Create game descriptions
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
    
    # Run a single test on the currently visible level
    bugs = pipeline._execute_test_task(game_description, feature_description, test_task)
    
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
