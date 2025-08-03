from vagen.env.base.base_env import BaseEnv
import ai2thor.controller
import ai2thor.platform
import ai2thor.fifo_server
import numpy as np
import time
import math
import os
import sys
import time

from vagen.env.utils.context_utils import convert_numpy_to_PIL
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from .env_config import SpocEnvConfig
from .prompt import system_prompt,init_observation_template, action_template, format_prompt
from .task_loader import get_dataset
from vagen.env.utils.state_reward_text_utils import env_state_reward_wrapper
from .spoc_fixes import (
    get_spoc_compatible_thor_config,
    get_spoc_initialize_params,
    is_object_type_match_spoc,
    validate_scene_name,
    get_wrist_rotation_delta,
    clamp_arm_position
)

class SpocEnv(BaseEnv):
    """SPOC environment with Stretch robot manipulation tasks."""   

    ValidEvalSets = [
        'base', 'common_sense', 'complex_instruction', 'visual_appearance', 'long_horizon'
    ]

    # Available actions - 20 actions for Stretch robot
    ACTION_LOOKUP = {
        # Navigation actions (8)
        "moveahead": 1,      # move_ahead
        "moveback": 2,       # move_back  
        "rotateright": 3,    # rotate_right
        "rotateleft": 4,     # rotate_left
        "rotateright_small": 5,  # rotate_right_small
        "rotateleft_small": 6,   # rotate_left_small
        "lookup": 7,         # deprecated for stretch
        "lookdown": 8,       # deprecated for stretch
        
        # Manipulation actions (12)
        "pickup": 9,         # pickup action
        "dropoff": 10,       # dropoff action
        "move_arm_up": 11,   # yp - move arm up 0.1m
        "move_arm_down": 12, # ym - move arm down 0.1m  
        "move_arm_out": 13,  # zp - extend arm out 0.1m
        "move_arm_in": 14,   # zm - retract arm in 0.1m
        "wrist_open": 15,    # wp - rotate wrist counterclockwise 10°
        "wrist_close": 16,   # wm - rotate wrist clockwise 10°
        "move_arm_up_small": 17,   # yps - move arm up 0.02m
        "move_arm_down_small": 18, # yms - move arm down 0.02m
        "move_arm_out_small": 19,  # zps - extend arm out 0.02m
        "move_arm_in_small": 20,   # zms - retract arm in 0.02m
    }

    # Action descriptions for Stretch robot
    DISCRETE_SKILLSET = [
        "Move the agent's base forward by 0.2 meters",          # moveahead
        "Move the agent's base backward by 0.2 meters",         # moveback  
        "Rotate the agent's base right by 30°",                 # rotateright
        "Rotate the agent's base left by 30°",                  # rotateleft
        "Rotate the agent's base right by 6°",                  # rotateright_small
        "Rotate the agent's base left by 6°",                   # rotateleft_small
        "Tilt navigation camera upward by 10°",                 # lookup
        "Tilt navigation camera downward by 10°",               # lookdown
        "Initiate a grasp action to pick up an object",         # pickup
        "Execute a release action to drop an object",           # dropoff
        "Move the arm up by 0.1 meters",                       # move_arm_up
        "Move the arm down by 0.1 meters",                     # move_arm_down
        "Extend the arm outward by 0.1 meters",                # move_arm_out
        "Retract the arm inward by 0.1 meters",                # move_arm_in
        "Rotate the wrist counterclockwise by 10°",            # wrist_open
        "Rotate the wrist clockwise by 10°",                   # wrist_close
        "Move the arm up by 0.02 meters",                      # move_arm_up_small
        "Move the arm down by 0.02 meters",                    # move_arm_down_small
        "Extend the arm outward by 0.02 meters",               # move_arm_out_small
        "Retract the arm inward by 0.02 meters",               # move_arm_in_small
    ]

    def __init__(self, config: SpocEnvConfig):
        """Initialize the SPOC environment."""
        super().__init__()
        self.config = config
        
        # Enhanced environment variables for better headless mode stability
        env_vars_to_set = {
            # 'DISPLAY': '',
            'XAUTHORITY': '',
            'XDG_RUNTIME_DIR': '/tmp',
            'GALLIUM_DRIVER': 'softpipe',
            'MESA_GL_VERSION_OVERRIDE': '3.3',
            'LIBGL_ALWAYS_SOFTWARE': '1',
            'LIBGL_ALWAYS_INDIRECT': '1',
            'EGL_PLATFORM': 'surfaceless',
            'PYOPENGL_PLATFORM': 'osmesa',  # Try OSMesa instead of EGL
            'MESA_NO_ERROR': '1',
        }
        for key, value in env_vars_to_set.items():
            os.environ[key] = value

        # Use SPOC-compatible configuration with correct commit ID
        self.thor_config = get_spoc_compatible_thor_config()
        # Add server_class which is not in the base config but needed
        self.thor_config["server_class"] = ai2thor.fifo_server.FifoServer

        self.env = None
        
        # Use SPOC official platform configuration logic
        try:
            # Determine platform and GPU device (following SPOC's logic)
            gpu_device = 0  # Default GPU device
            
            # SPOC platform selection logic
            if sys.platform.lower() == "darwin":
                platform = ai2thor.platform.OSXIntel64
                controller_args = self.thor_config.copy()
            else:
                # Use CloudRendering for non-macOS systems
                platform = ai2thor.platform.CloudRendering
                controller_args = {
                    **self.thor_config,
                    "platform": platform,
                    "gpu_device": gpu_device,
                }
            
            print(f"[SPOC] Initializing AI2-THOR with platform: {platform}")
            print(f"[SPOC] Controller args: server_timeout={controller_args.get('server_timeout')}, commit_id={controller_args.get('commit_id')}")
            
            # Initialize controller with SPOC configuration
            self.env = ai2thor.controller.Controller(**controller_args)
            
            # Verify the commit ID matches SPOC requirements
            if hasattr(self.env, '_build') and self.env._build and hasattr(self.env._build, 'commit_id'):
                actual_commit = self.env._build.commit_id
                expected_commit = self.thor_config["commit_id"]  # Use the correct commit from config
                if expected_commit not in actual_commit:
                    print(f"[WARNING] AI2-THOR commit mismatch. Expected: {expected_commit}, Got: {actual_commit}")
                else:
                    print(f"[SUCCESS] AI2-THOR commit verified: {actual_commit}")
            
            print(f"[SUCCESS] AI2-THOR initialized successfully with SPOC configuration")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize AI2-THOR with SPOC configuration: {e}")
            print(f"[ERROR] This may be due to AI2-THOR version mismatch or missing CloudRendering support")
            raise RuntimeError(f"Failed to initialize AI2-THOR with SPOC configuration: {e}")
        
        # --- Dataset Loading ---
        self.dataset = get_dataset(
            data_path=config.data_path, 
            task_type=config.task_type, 
            split=config.chores_split
        )
        print(f"[SpocEnv] Loaded {len(self.dataset)} episodes")
        self.number_of_episodes = len(self.dataset)
        
        # --- Episode State Tracking ---
        self._current_episode_num = 0
        self._current_step = 0
        self._max_episode_steps = 100 # Increased for manipulation tasks
        self._episode_start_time = 0
        self.is_holding = False
        self.episode_log = []
        self.episode_language_instruction = ""
        self.episode_data = None
        self._last_event = None
        self.total_reward = 0
        self.valid_actions = []
        self.reward = 0
        self.agent_path = []  # Track agent positions for path visualization
        
        # --- Initialize third party camera setup for manipulation camera ---
        self._setup_manipulation_camera()
        
        # --- New reward tracking variables ---
        self.prev_distance_to_target = None
        self.prev_holding = False
        self.prev_position = None
        self.action_history = []
        self.exploration_positions = set()
        self.last_pickup_attempt_step = -10  # Track pickup attempts
        
        # Action success tracking
        self._last_action_success = True
        self._last_action_failure_reason = ""
        
        self.format_prompt_func = format_prompt[self.config.prompt_format]
        self.parse_func = PARSE_FUNC_MAP[self.config.prompt_format]
    
    def _setup_manipulation_camera(self):
        """Setup manipulation camera as third party camera like SPOC official implementation."""
        try:
            # SPOC uses thirdPartyCameras[0] for manipulation camera
            # This will be set up during reset/calibration, but we prepare the groundwork here
            self._manipulation_camera_initialized = False
            print("[SPOC Camera] Manipulation camera setup prepared")
        except Exception as e:
            print(f"[WARNING] Failed to prepare manipulation camera setup: {e}")
    
    @property
    def navigation_camera(self):
        """Get navigation camera with SPOC-compatible cutoff like official implementation."""
        try:
            frame = self.env.last_event.frame
            if frame is not None:
                cutoff = round(frame.shape[1] * 6 / 396)
                return frame[:, cutoff:-cutoff, :]
            else:
                return None
        except Exception as e:
            print(f"[ERROR] Failed to get navigation camera: {e}")
            return None
    
    @property 
    def manipulation_camera(self):
        """Get manipulation camera with SPOC-compatible cutoff like official implementation."""
        try:
            if len(self.env.last_event.third_party_camera_frames) > 0:
                frame = self.env.last_event.third_party_camera_frames[0]
                cutoff = round(frame.shape[1] * 6 / 396)
                return frame[:, cutoff:-cutoff, :3]  # Only RGB channels like SPOC
            else:
                return None
        except Exception as e:
            print(f"[ERROR] Failed to get manipulation camera: {e}")
            return None
        
    def reset(self, seed=None):
        """Reset the environment to a new episode from the real dataset."""
        if self.number_of_episodes == 0:
            raise RuntimeError("Cannot reset environment, SPOC dataset is empty or not found.")

        # Use a deterministic or random index based on the seed
        if seed is not None:
            # Use a random number generator seeded for reproducibility
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, self.number_of_episodes)
        else:
            idx = np.random.randint(0, self.number_of_episodes)
        
        self._current_episode_num = idx
        
        # Load real episode data using our new task loader
        traj_data = self.dataset[idx]
        self.episode_data = traj_data
        
        # Enhance the instruction with specific target information
        base_instruction = traj_data["instruction"]
        target_type = traj_data.get("targetObjectType", "unknown object")
        
        # Debug: Check what target_type we got
        print(f"[DEBUG EPISODE] Episode {idx}: targetObjectType='{target_type}', instruction='{base_instruction}'")
        
        # Create more specific instruction - handle None/empty cases
        if target_type and target_type != "unknown object" and target_type != "None" and str(target_type).strip():
            self.episode_language_instruction = f"Find and fetch a {target_type}. Navigate around the environment to locate the {target_type}, then approach it and pick it up."
            print(f"[DEBUG EPISODE] Using enhanced instruction: {self.episode_language_instruction}")
        else:
            # Fallback: try to extract from original instruction
            if "fetch" in base_instruction.lower() or "get" in base_instruction.lower():
                self.episode_language_instruction = "Find and fetch the target object. Navigate around the environment to locate objects you can pick up."
            else:
                self.episode_language_instruction = base_instruction
            print(f"[DEBUG EPISODE] Using fallback instruction: {self.episode_language_instruction}")

        # Reset the AI2-THOR scene with validation
        scene_data = traj_data["scene"]
        house_index = traj_data.get("house_index", "unknown")
        
        # Check if we have ProcTHOR scene JSON or AI2-THOR scene name
        if isinstance(scene_data, dict):
            # ProcTHOR scene JSON format
            print(f"[SPOC] Using ProcTHOR scene for house_index {house_index}")
            scene_json = scene_data
            scene_name = f"ProcTHOR_House_{house_index}"
        else:
            # Legacy AI2-THOR FloorPlan format (fallback)
            print(f"[SPOC] Warning: Using legacy AI2-THOR FloorPlan format: {scene_data}")
            scene_name = scene_data
            # Validate scene name and correct if needed
            is_valid, corrected_scene = validate_scene_name(scene_name)
            if not is_valid:
                print(f"[SPOC] Warning: Scene {scene_name} not valid, using {corrected_scene}")
                scene_name = corrected_scene
            scene_json = None
            
        max_retries = 3
        reset_success = False
        for attempt in range(max_retries):
            try:
                if scene_json is not None:
                    # Step 1: Reset with ProcTHOR scene JSON (like official SPOC)
                    print(f"[DEBUG] Attempting to reset to ProcTHOR scene: {scene_name}")
                    self._last_event = self.env.reset(scene=scene_json)
                else:
                    # Step 1: Reset the AI2-THOR scene using standard iTHOR reset (fallback)
                    print(f"[DEBUG] Attempting to reset to AI2-THOR scene: {scene_name}")
                    self._last_event = self.env.reset(scene=scene_name)
                if not self._last_event or not self._last_event.metadata.get('lastActionSuccess'):
                    raise RuntimeError(f"Attempt {attempt + 1}: Failed to reset to scene {scene_name}.")
                
                # Step 1.5: Apply SPOC initialization parameters (especially visibilityDistance)
                init_params = get_spoc_initialize_params()
                print(f"[SPOC] Applying initialize parameters: {init_params}")
                if init_params:  # Only call Initialize if there are parameters
                    print(f"[DEBUG] Calling Initialize with params: {init_params}")
                    init_event = self.env.step(action="Initialize", **init_params)
                    print(f"[DEBUG] Initialize completed")
                    if not init_event.metadata.get('lastActionSuccess'):
                        print(f"[SPOC] Warning: Initialize action failed: {init_event.metadata.get('errorMessage', 'Unknown error')}")
                        # Don't raise error - this is not critical
                else:
                    print(f"[DEBUG] Skipping Initialize action (no parameters)")
                
                # Step 2: Teleport the agent to the starting pose
                if scene_json is not None:
                    # For ProcTHOR scenes, the agent position from dataset should be used
                    # But we need to be careful - the scene might have its own agent position
                    pose = traj_data["agentPose"]
                    print(f"[DEBUG] About to teleport agent in ProcTHOR scene to position: {pose['position']}, rotation: {pose['rotation']}")
                    
                    # Use the dataset pose (from the original trajectory)
                    self._last_event = self.env.step(
                        action="TeleportFull",
                        position=pose["position"],
                        rotation={'x': 0, 'y': pose["rotation"], 'z': 0},
                        horizon=0,  # Use HORIZON=0 as in SPOC
                        standing=True,
                        forceAction=True
                    )
                else:
                    # For AI2-THOR FloorPlan scenes (fallback)
                    pose = traj_data["agentPose"]
                    print(f"[DEBUG] About to teleport agent in AI2-THOR scene to position: {pose['position']}, rotation: {pose['rotation']}")
                    self._last_event = self.env.step(
                        action="TeleportFull",
                        position=pose["position"],
                        rotation={'x': 0, 'y': pose["rotation"], 'z': 0},
                        horizon=0,  # Use HORIZON=0 as in SPOC
                        standing=True,
                        forceAction=True
                    )
                print(f"[DEBUG] Teleport completed, success: {self._last_event.metadata.get('lastActionSuccess') if self._last_event else 'No event'}")
                if not self._last_event or not self._last_event.metadata.get('lastActionSuccess'):
                    raise RuntimeError(f"Attempt {attempt + 1}: Failed to teleport agent in scene {scene_name}.")
                
                # Step 2.5: Calibrate agent like SPOC (cameras, gripper, etc.)
                print(f"[DEBUG] About to calibrate agent...")
                self._calibrate_agent()
                print(f"[DEBUG] Agent calibration completed")

                # Step 3: Enable lighting in the scene
                try:
                    # First try to randomize lighting to ensure proper illumination
                    lighting_event = self.env.step(action="RandomizeLighting", brightness=(0.5, 1.5), randomizeColor=False)
                    if not lighting_event.metadata.get('lastActionSuccess'):
                        print(f"Warning: RandomizeLighting failed for scene {scene_name}")
                    
                    # Additionally, try to turn on any lamps/lights in the scene
                    visible_objects = self.env.last_event.metadata.get('objects', [])
                    for obj in visible_objects:
                        if obj.get('objectType', '').lower() in ['floorlamp', 'desklamp', 'lamp'] and not obj.get('isToggled', False):
                            try:
                                toggle_event = self.env.step(action="ToggleObjectOn", objectId=obj['objectId'])
                                if toggle_event.metadata.get('lastActionSuccess'):
                                    print(f"Successfully turned on light: {obj['objectId']}")
                            except Exception as light_error:
                                # Don't fail the whole reset if individual lights can't be turned on
                                pass
                                
                except Exception as lighting_error:
                    print(f"Warning: Failed to set up lighting in scene {scene_name}: {lighting_error}")
                    # Don't fail the whole reset due to lighting issues

                # If both steps succeed, mark as successful and break the loop
                reset_success = True
                break

            except Exception as e:
                print(f"Warning: SpocEnv.reset failed on attempt {attempt + 1}/{max_retries} for scene '{scene_name}'. Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait for 2 seconds before retrying
                else:
                    # If all retries fail, DO NOT raise an error.
                    # Instead, print a critical warning and try to reset with a NEW random episode.
                    print(f"CRITICAL: Scene '{scene_name}' seems to be broken or has invalid coordinates. "
                        f"Skipping this episode and trying a new one.")
                    return self.reset(seed=None)  # Recursively call reset to get a new valid episode

        # Reset episode tracking information
        self._current_step = 0
        self.is_holding = False
        self.episode_log = []
        self._episode_start_time = time.time()
        self.total_reward = 0
        self.valid_actions = []
        self.reward = 0
        self.agent_path = [pose["position"]]  # Initialize with starting position
        
        # Reset reward tracking variables
        self.prev_distance_to_target = None
        self.prev_holding = False
        self.prev_position = None
        self.action_history = []
        self.exploration_positions = set()
        self.last_pickup_attempt_step = -10
        
        # Initialize info dict with metrics for consistency
        info = {
            "metrics": {
                "turn_metrics": {
                    "action_is_valid": False,
                    "action_is_effective": False,
                },
                "traj_metrics": {
                    "success": False,
                }
            },
            "env_feedback": "Environment reset successfully.",
            "is_format_rewarded": False,
            "distance": 0.0,
            "instruction": self.episode_language_instruction,
            "env_step": 0,
            "episode_elapsed_seconds": 0.0,
            "task_success": False,
            "last_action_success": True,
            "llm_raw_response": ""
        }
        return self._render(init_obs=True), info
    
    @env_state_reward_wrapper
    def step(self, action_str: str):
        """Execute an action in the environment.
        
        This method:
        1. Parses the raw LLM response to extract actions
        2. Executes each valid action in sequence
        3. Calculates rewards and metrics
        4. Generates the next observation
        
        Args:
            action_str: Raw text response from LLM
            
        Returns:
            Observation, reward, done, info
        """
        # Process the LLM response to extract actions
        print(f"[DEBUG SPOC] ===== Starting step() with action_str =====")
        print(f"[DEBUG SPOC] Raw LLM response: {action_str}")
        # print(f"[DEBUG SPOC] Response length: {len(action_str)} chars")
        
        rst = self.parse_func(
            response=action_str,
            special_token_list=getattr(self.config, 'special_token_list', None),
            action_sep=getattr(self.config, 'action_sep', ',') or ',',
            max_actions=getattr(self.config, 'max_actions_per_step', 1) or 1
        )
        
        # print(f"[DEBUG SPOC] Parse result: {rst}")
        # print(f"[DEBUG SPOC] Format correct: {rst.get('format_correct', False)}")
        # print(f"[DEBUG SPOC] Actions extracted: {rst.get('actions', [])}")
        
        action_list = rst['actions']
        prev_pos = self.env.last_event.metadata["agent"]["position"]
        
        metrics = {
            "turn_metrics": {
                "action_is_valid": len(action_list) > 0,
                "action_is_effective": False,
            },
            "traj_metrics": {
                "success": False,
            }
        }
        
        self.reward = 0
        self.valid_actions = []
        done = False

        info = {}
        info.update(rst)
        # 重要：添加llm_raw_response键，rollout_manager需要它来构建prompt
        info['llm_raw_response'] = action_str
            
            
        # Execute valid actions
        print(f"[DEBUG SPOC] action_is_valid: {metrics['turn_metrics']['action_is_valid']}")
        print(f"[DEBUG SPOC] format_correct: {rst.get('format_correct', True)}")
        
        if metrics["turn_metrics"]["action_is_valid"] and rst.get("format_correct", True):
            print(f"[DEBUG SPOC] Executing {len(action_list)} actions: {action_list}")
            
            for i, action in enumerate(action_list):
                action_lower = action.lower()
                print(f"[DEBUG SPOC] Processing action {i+1}/{len(action_list)}: '{action}' -> '{action_lower}'")
                
                if action_lower in self.ACTION_LOOKUP:
                    action_int = self.ACTION_LOOKUP[action_lower]
                    print(f"[DEBUG SPOC] Valid action found: {action_lower} -> {action_int}")
                    self._execute_action(action_int)
                    
                    # Track agent path for map visualization
                    curr_agent_pos = self.env.last_event.metadata["agent"]["position"]
                    self.agent_path.append(curr_agent_pos)
                    
                    success, distance = self.measure_success()
                    
                    # Check for task success
                    success, distance = self.measure_success()
                    if success:
                        done = True
                        metrics['traj_metrics']['success'] = True
                        print(f"[DEBUG SPOC] SUCCESS! Task completed with action: {action_lower}")
                    
                    self.valid_actions.append(action)
                    
                    if done:
                        break
                else:
                    print(f"[DEBUG SPOC] INVALID ACTION: '{action_lower}' not in ACTION_LOOKUP")
                    print(f"[DEBUG SPOC] Available actions: {list(self.ACTION_LOOKUP.keys())}")
                    metrics['turn_metrics']['action_is_valid'] = False
                    break
                
                self._current_step += 1
                if self._current_step >= self._max_episode_steps:
                    done = True
                    break
        else:
            print(f"[DEBUG SPOC] SKIPPING action execution - invalid format or no valid actions")
        
        # Check if the agent position has changed (action was effective)
        curr_pos = self.env.last_event.metadata["agent"]["position"]
        metrics['turn_metrics']['action_is_effective'] = curr_pos["x"] != prev_pos["x"] or curr_pos["z"] != prev_pos["z"]
        
        # Compute comprehensive reward using new system
        step_reward, reward_breakdown = self._compute_step_reward(action_list, metrics, rst, prev_pos, curr_pos)
        self.reward = step_reward
        info["reward_breakdown"] = reward_breakdown
        info["is_format_rewarded"] = reward_breakdown.get('format', 0.0) > 0
        
        # Update info dict
        info["metrics"] = metrics
        success, distance = self.measure_success()
        info['distance'] = distance
        info['instruction'] = self.episode_language_instruction
        info['env_step'] = self._current_step
        info['episode_elapsed_seconds'] = time.time() - self._episode_start_time
        info['task_success'] = success
        info['last_action_success'] = self._last_action_success
        
        # Generate detailed feedback based on action success
        if len(action_list) > 0:
            action_name = action_list[0].lower()
            if self._last_action_success:
                info["env_feedback"] = f"Action '{action_name}' executed successfully."
            else:
                info["env_feedback"] = f"Action '{action_name}' FAILED - {self._last_action_failure_reason}. Try a different approach."
        else:
            info["env_feedback"] = "No valid action was provided. Please specify a valid action."
        self.info = info
        # Update total reward
        self.total_reward += self.reward
        
        return self._render(init_obs=False), self.reward, done, info
    
    def _execute_action(self, action_index: int):
        """Executes a discrete action for the Stretch robot using SPOC-compatible actions."""
        # Store state before action for validation
        before_state = {
            "position": self.env.last_event.metadata["agent"]["position"].copy(),
            "rotation": self.env.last_event.metadata["agent"]["rotation"].copy(),
            "arm_state": self._get_current_arm_state().copy(),
            "holding": self.is_holding
        }
        
        # Get current arm state for calculating absolute positions
        current_arm_state = self._get_current_arm_state()
        
        # Base action map for non-arm/wrist actions
        action_map = {
            # Navigation actions - use SPOC-compatible MoveAgent
            1: {"action": "MoveAgent", "ahead": 0.2},
            2: {"action": "MoveAgent", "ahead": -0.2},
            # Rotation actions - use SPOC-compatible RotateAgent
            3: {"action": "RotateAgent", "degrees": 30},
            4: {"action": "RotateAgent", "degrees": -30},
            5: {"action": "RotateAgent", "degrees": 6},
            6: {"action": "RotateAgent", "degrees": -6},
            # Camera pitch actions - Stretch CAN do this via RotateCameraMount
            7: {"action": "RotateCameraMount", "degrees": 10, "secondary": False},  # lookup
            8: {"action": "RotateCameraMount", "degrees": -10, "secondary": False}, # lookdown
            # Object manipulation
            9: {"action": "PickupObject"},
            10: {"action": "ReleaseObject"},
        }
        
        # Handle arm movements with bounds checking
        if action_index in range(11, 21) and action_index not in [15, 16]:
            if action_index == 11:  # move_arm_up
                new_pos = {"x": current_arm_state["x"], "y": current_arm_state["y"] + 0.1, "z": current_arm_state["z"]}
            elif action_index == 12:  # move_arm_down
                new_pos = {"x": current_arm_state["x"], "y": current_arm_state["y"] - 0.1, "z": current_arm_state["z"]}
            elif action_index == 13:  # move_arm_out
                new_pos = {"x": current_arm_state["x"], "y": current_arm_state["y"], "z": current_arm_state["z"] + 0.1}
            elif action_index == 14:  # move_arm_in
                new_pos = {"x": current_arm_state["x"], "y": current_arm_state["y"], "z": current_arm_state["z"] - 0.1}
            elif action_index == 17:  # move_arm_up_small
                new_pos = {"x": current_arm_state["x"], "y": current_arm_state["y"] + 0.02, "z": current_arm_state["z"]}
            elif action_index == 18:  # move_arm_down_small
                new_pos = {"x": current_arm_state["x"], "y": current_arm_state["y"] - 0.02, "z": current_arm_state["z"]}
            elif action_index == 19:  # move_arm_out_small
                new_pos = {"x": current_arm_state["x"], "y": current_arm_state["y"], "z": current_arm_state["z"] + 0.02}
            elif action_index == 20:  # move_arm_in_small
                new_pos = {"x": current_arm_state["x"], "y": current_arm_state["y"], "z": current_arm_state["z"] - 0.02}
            
            # Clamp position to bounds
            new_pos = clamp_arm_position(new_pos)
            action_map[action_index] = {"action": "MoveArm", "position": new_pos}
        
        # Handle wrist rotation with bounds
        elif action_index in [15, 16]:
            try:
                arm_metadata = self.env.last_event.metadata['arm']
                joints = arm_metadata['joints']
                # Find wrist joint rotation
                wrist_joint = None
                for joint in joints:
                    if joint["name"] == "stretch_robot_wrist_2_jnt":
                        wrist_joint = joint
                        break
                
                if wrist_joint:
                    current_wrist = wrist_joint["rootRelativeRotation"]["w"] * wrist_joint["rootRelativeRotation"]["y"]
                    
                    if action_index == 15:  # wrist_open
                        yaw = get_wrist_rotation_delta(current_wrist, "open")
                    else:  # wrist_close
                        yaw = get_wrist_rotation_delta(current_wrist, "close")
                    
                    action_map[action_index] = {"action": "RotateWristRelative", "yaw": yaw}
                else:
                    # Fallback if can't find wrist joint
                    action_map[15] = {"action": "RotateWristRelative", "yaw": -10}
                    action_map[16] = {"action": "RotateWristRelative", "yaw": 10}
            except:
                # Fallback
                action_map[15] = {"action": "RotateWristRelative", "yaw": -10}
                action_map[16] = {"action": "RotateWristRelative", "yaw": 10}

        params = action_map.get(action_index, {"action": "Pass"})
        
        # Add SPOC-compatible additional arguments
        if params["action"] in ["MoveArm", "RotateWristRelative"]:
            # Add SPOC's ADDITIONAL_ARM_ARGS equivalent
            params.update({
                "disableRendering": True,
                "returnToStart": True,
                "speed": 1
            })
        elif params["action"] == "MoveAgent":
            # Add SPOC's ADDITIONAL_NAVIGATION_ARGS equivalent  
            params.update({
                "disableRendering": True,
                "returnToStart": True
            })
        
        self._last_event = self.env.step(**params)

        # Update held status using proper AI2-THOR metadata (not just a flag)
        self._update_holding_status()
        
        # Add physics step for dropoff action like SPOC
        if params["action"] == "ReleaseObject":
            self.env.step(action="AdvancePhysicsStep", simSeconds=2)
        
        # Validate action success by checking if state actually changed
        after_state = {
            "position": self.env.last_event.metadata["agent"]["position"].copy(),
            "rotation": self.env.last_event.metadata["agent"]["rotation"].copy(),
            "arm_state": self._get_current_arm_state().copy(),
            "holding": self.is_holding
        }
        
        # Check action success based on action type (following SPOC's logic)
        action_success = True
        action_failure_reason = ""
        
        if action_index in [1, 2]:  # Movement actions
            position_change = math.sqrt(
                (after_state["position"]["x"] - before_state["position"]["x"])**2 +
                (after_state["position"]["z"] - before_state["position"]["z"])**2
            )
            if position_change < 0.01:  # SPOC's threshold
                action_success = False
                action_failure_reason = "collision - cannot move in that direction"
                
        elif action_index in [3, 4, 5, 6]:  # Rotation actions
            rotation_change = abs(after_state["rotation"]["y"] - before_state["rotation"]["y"])
            # Handle wrap-around
            if rotation_change > 180:
                rotation_change = 360 - rotation_change
            if rotation_change < 2:  # SPOC's threshold
                action_success = False
                action_failure_reason = "collision - cannot rotate"
                
        elif action_index in range(11, 21) and action_index not in [15, 16]:  # Arm movements
            arm_change = math.sqrt(
                (after_state["arm_state"]["x"] - before_state["arm_state"]["x"])**2 +
                (after_state["arm_state"]["y"] - before_state["arm_state"]["y"])**2 +
                (after_state["arm_state"]["z"] - before_state["arm_state"]["z"])**2
            )
            if arm_change < 0.001:  # SPOC's threshold
                action_success = False
                action_failure_reason = "arm at limit or collision"
                
        elif action_index == 9:  # Pickup
            if not self.is_holding and before_state["holding"] == self.is_holding:
                action_success = False
                action_failure_reason = "no object within reach to pick up"
                
        elif action_index == 10:  # Dropoff
            if self.is_holding and before_state["holding"] == self.is_holding:
                action_success = False
                action_failure_reason = "failed to release object"
        
        # Store action success info for feedback
        self._last_action_success = action_success
        self._last_action_failure_reason = action_failure_reason

    def _get_current_arm_state(self):
        """Get current arm state for absolute positioning like SPOC."""
        try:
            arm_metadata = self.env.last_event.metadata['arm']
            joints = arm_metadata['joints']
            
            # Calculate relative arm position like SPOC's get_relative_stretch_current_arm_state
            z = joints[-1]["rootRelativePosition"]["z"]
            x = joints[-1]["rootRelativePosition"]["x"]  
            y = joints[0]["rootRelativePosition"]["y"] - 0.16297650337219238  # SPOC's baseline offset
            
            return {"x": x, "y": y, "z": z}
        except (KeyError, IndexError):
            # Fallback to default retracted position
            return {"x": 0.0, "y": 0.8, "z": 0.0}
    
    def _update_holding_status(self):
        """Update holding status based on actual AI2-THOR metadata."""
        try:
            held_objects = self.env.last_event.metadata['arm']['heldObjects']
            self.is_holding = len(held_objects) > 0
        except (KeyError, IndexError):
            self.is_holding = False

    def measure_success(self):
        """
        Check if the agent has successfully completed the Fetch task.
        Success is defined as holding an object of the correct type using SPOC-compatible matching.
        """
        # Default to not successful
        success = False
        distance = float('inf')
        
        try:
            # Always get agent position for distance calculation
            agent_pos = self.env.last_event.metadata["agent"]["position"]
            
            # Check if we are holding the correct object type using proper AI2-THOR metadata
            held_objects = self.env.last_event.metadata['arm']['heldObjects']
            target_type = self.episode_data.get("targetObjectType") if self.episode_data else None
            
            if held_objects and target_type:
                # Check if any held object matches the target type using SPOC-compatible matching
                for obj in held_objects:
                    if self._is_object_type_match(obj['objectType'], target_type):
                        success = True
                        print(f"[SUCCESS] Successfully holding target object: {obj['objectType']} matches {target_type}")
                        break
            
            # Calculate distance to target (either held object or closest target in scene)
            if success and held_objects:
                # If holding target, distance is 0
                distance = 0.0
            else:
                # Find closest target object in scene for distance calculation
                distance = self._calculate_distance_to_closest_target(agent_pos, target_type)
                
        except (KeyError, IndexError) as e:
            # Metadata might not be available if the last action failed
            print(f"Warning: Could not check success due to missing metadata: {e}")
            success = False
            # Fallback distance calculation
            if self.episode_data and self.episode_data.get("target_position"):
                target_pos = self.episode_data["target_position"]
                distance = math.sqrt(
                    (agent_pos["x"] - target_pos["x"])**2 +
                    (agent_pos["z"] - target_pos["z"])**2
                )
        
        return float(success), distance
    
    def _prepare_scene_with_navigation_mesh(self, scene_name: str) -> dict:
        """
        Prepare scene data with SPOC-compatible navigation mesh setup.
        Based on SPOC's StretchController.reset() method.
        """
        # SPOC uses multiple agent radii for robust path planning
        AGENT_RADIUS_LIST = [
            (0, 0.18),  # Most restrictive - tight spaces
            (1, 0.225), # Medium restrictive  
            (2, 0.3)    # Least restrictive - open areas
        ]
        
        # Base navigation mesh configuration from SPOC
        base_agent_navmesh = {
            "agentHeight": 1.8,
            "agentSlope": 10,
            "agentClimb": 0.5,
            "voxelSize": 0.1666667,
        }
        
        # Create scene data with navigation meshes
        scene_data = {
            "sceneName": scene_name,
            "metadata": {
                "agent": {
                    "horizon": 0,  # SPOC uses HORIZON=0
                    "position": {"x": 0, "y": 0.95, "z": 0},  # Default position
                    "rotation": {"x": 0, "y": 270, "z": 0},   # Default rotation
                    "standing": True,
                },
                "navMeshes": [
                    {**base_agent_navmesh, **{"id": i, "agentRadius": r}} 
                    for (i, r) in AGENT_RADIUS_LIST
                ]
            }
        }
        
        return scene_data
        
    def _calibrate_agent(self):
        """
        Calibrate the agent like SPOC's calibrate_agent() method.
        Sets up cameras, gripper, and other agent parameters.
        """
        try:
            # Step 1: Ensure agent is in standing position with horizon=0
            self.env.step(action="Teleport", horizon=0, standing=True)
            
            # Step 2: Calibrate navigation camera mount (with SPOC's randomization)
            nav_camera_angle = 27.0 + np.random.uniform(-2, 2)  # SPOC's ±2° randomization
            self.env.step(
                action="RotateCameraMount",
                degrees=nav_camera_angle,
                secondary=False  # Navigation camera
            )
            
            # Step 3: Set navigation camera FOV (with SPOC's randomization)
            nav_fov = 59 + np.random.uniform(-1, 1)  # SPOC's ±1° randomization
            self.env.step(
                action="ChangeFOV",
                fieldOfView=nav_fov,
                camera="FirstPersonCharacter"
            )
            
            # Step 4: Calibrate manipulation camera mount (with SPOC's randomization)
            manip_camera_angle = 33.0 + np.random.uniform(-2, 2)  # SPOC's ±2° randomization
            self.env.step(
                action="RotateCameraMount",
                degrees=manip_camera_angle,
                secondary=True  # Manipulation camera
            )
            
            # Step 5: Set manipulation camera FOV (with SPOC's randomization)
            manip_fov = 59 + np.random.uniform(-1, 1)  # SPOC's ±1° randomization
            self.env.step(
                action="ChangeFOV",
                fieldOfView=manip_fov,
                camera="SecondaryCamera"
            )
            
            # Step 6: Set gripper openness to SPOC's default (30°)
            self.env.step(action="SetGripperOpenness", openness=30)
            
            # Step 7: Hide the unrealistic blue sphere on the gripper (like SPOC)
            self.env.step("ToggleMagnetVisibility", visible=False, raise_for_failure=True)
            
            # Step 8: Initialize manipulation camera (third party camera) like SPOC
            self._initialize_manipulation_camera()
            
            print("[SPOC Calibration] Agent calibration completed successfully")
            
        except Exception as e:
            print(f"[WARNING] Agent calibration failed: {e}")
            # Don't fail the whole reset if calibration has issues
    
    def _initialize_manipulation_camera(self):
        """Initialize manipulation camera as third party camera like SPOC official does."""
        try:
            # Check if we already have the required third party camera
            if len(self.env.last_event.third_party_camera_frames) == 0:
                # SPOC ensures thirdPartyCameras[0] exists during initialization
                # The manipulation camera should already be set up by AI2-THOR's Stretch agent setup
                # We just need to verify it's working
                self.env.step(action="Pass")  # Refresh to ensure cameras are active
                
                if len(self.env.last_event.third_party_camera_frames) > 0:
                    self._manipulation_camera_initialized = True
                    print("[SPOC Camera] Manipulation camera (third party camera) initialized successfully")
                else:
                    print("[WARNING] Manipulation camera not available - will use navigation camera for both views")
                    self._manipulation_camera_initialized = False
            else:
                self._manipulation_camera_initialized = True
                print("[SPOC Camera] Manipulation camera already available")
                
        except Exception as e:
            print(f"[WARNING] Failed to initialize manipulation camera: {e}")
            self._manipulation_camera_initialized = False
    
    def _is_object_type_match(self, object_type: str, target_type: str) -> bool:
        """
        SPOC-compatible object type matching using the robust matching from spoc_fixes.
        """
        return is_object_type_match_spoc(object_type, target_type)
        
    def _calculate_distance_to_closest_target(self, agent_pos: dict, target_type: str) -> float:
        """
        Calculate distance to the closest target object in the scene.
        Uses SPOC-compatible 3D distance calculation.
        """
        if not target_type:
            return float('inf')
            
        try:
            objects = self.env.last_event.metadata.get("objects", [])
            min_distance = float('inf')
            
            for obj in objects:
                if obj.get("visible", False) and self._is_object_type_match(obj["objectType"], target_type):
                    obj_pos = obj["position"]
                    # Use SPOC's 3D position distance calculation
                    distance = math.sqrt(
                        (agent_pos["x"] - obj_pos["x"])**2 +
                        (agent_pos["y"] - obj_pos["y"])**2 +  # Include Y dimension like SPOC
                        (agent_pos["z"] - obj_pos["z"])**2
                    )
                    min_distance = min(min_distance, distance)
                    
            return min_distance if min_distance != float('inf') else 2.0  # Default distance if no targets found
            
        except Exception as e:
            print(f"Warning: Error calculating target distance: {e}")
            return 2.0
    
    def _compute_step_reward(self, action_list, metrics, rst, prev_pos, curr_pos):
        """Compute comprehensive step reward with multiple components."""
        reward = 0.0
        reward_breakdown = {}
        
        # 1. Format reward - restore higher weight for initial learning
        if metrics['turn_metrics']['action_is_valid'] and rst.get('format_correct', True):
            format_reward = self.config.format_reward * 0.5  # Increased from 0.1 to 0.5 for better initial success
            reward += format_reward
            reward_breakdown['format'] = format_reward
        else:
            reward_breakdown['format'] = 0.0
            
        # 2. Task success reward - highest priority
        success, current_distance = self.measure_success()
        if success:
            success_reward = 10.0
            reward += success_reward
            reward_breakdown['success'] = success_reward
        else:
            reward_breakdown['success'] = 0.0
            
        # 3. Distance-based progress reward - more generous for initial learning
        if self.prev_distance_to_target is not None and current_distance is not None:
            distance_improvement = self.prev_distance_to_target - current_distance
            if distance_improvement > 0.05:  # Lower threshold for easier initial progress (was 0.1)
                progress_reward = min(distance_improvement * 3.0, 2.0)  # Higher reward multiplier and cap
                reward += progress_reward
                reward_breakdown['progress'] = progress_reward
            elif distance_improvement < -0.05:  # Moving away penalty with lower threshold
                retreat_penalty = max(distance_improvement * 0.5, -0.2)  # Reduced penalty for exploration
                reward += retreat_penalty
                reward_breakdown['progress'] = retreat_penalty
            else:
                reward_breakdown['progress'] = 0.0
        else:
            reward_breakdown['progress'] = 0.0
            
        # 4. Action effectiveness reward - increased for better initial learning
        if metrics['turn_metrics']['action_is_effective']:
            effectiveness_reward = 0.5  # Increased from 0.2 to 0.5
            reward += effectiveness_reward
            reward_breakdown['effectiveness'] = effectiveness_reward
        else:
            reward_breakdown['effectiveness'] = 0.0
            
        # 4.5. Failed action penalty - encourage trying different approaches
        if not self._last_action_success:
            failed_action_penalty = -0.5  # Significant penalty for failed actions
            reward += failed_action_penalty
            reward_breakdown['action_failure'] = failed_action_penalty
        else:
            reward_breakdown['action_failure'] = 0.0
            
        # 5. Object manipulation rewards
        current_holding = self.is_holding
        if current_holding and not self.prev_holding:
            # Check if we picked up the correct target object
            success, distance = self.measure_success()
            if success:
                pickup_reward = 10.0  # Big reward for picking up correct object!
                print(f"[BIG REWARD] 🎉 PICKED UP CORRECT TARGET! Reward: +{pickup_reward}")
            else:
                pickup_reward = 3.0  # Smaller reward for wrong object
                print(f"[REWARD] Picked up object (not target), Reward: +{pickup_reward}")
            reward += pickup_reward
            reward_breakdown['pickup'] = pickup_reward
        else:
            reward_breakdown['pickup'] = 0.0
            
        # 6. Exploration reward - encourage visiting new areas
        current_pos_key = (round(curr_pos['x'], 1), round(curr_pos['z'], 1))
        if current_pos_key not in self.exploration_positions:
            self.exploration_positions.add(current_pos_key)
            exploration_reward = 0.1
            reward += exploration_reward
            reward_breakdown['exploration'] = exploration_reward
        else:
            reward_breakdown['exploration'] = 0.0
            
        # 7. Repetitive action penalty
        if len(action_list) > 0:
            recent_actions = self.action_history[-5:]  # Look at last 5 actions
            repeated_count = sum(1 for a in recent_actions if a == action_list[0])
            if repeated_count >= 3:
                repetition_penalty = -0.1 * (repeated_count - 2)
                reward += repetition_penalty
                reward_breakdown['repetition'] = repetition_penalty
            else:
                reward_breakdown['repetition'] = 0.0
                
        # 8. Pickup attempt timing reward/penalty
        if len(action_list) > 0 and 'pickup' in action_list[0].lower():
            if current_distance and current_distance < 2.0:  # Close enough to attempt pickup
                if self._current_step - self.last_pickup_attempt_step > 5:  # Not too frequent
                    pickup_attempt_reward = 1.0  # Increased from 0.5 to 1.0
                    reward += pickup_attempt_reward
                    reward_breakdown['pickup_attempt'] = pickup_attempt_reward
                    self.last_pickup_attempt_step = self._current_step
                else:
                    pickup_spam_penalty = -0.1  # Reduced penalty from -0.2 to -0.1
                    reward += pickup_spam_penalty
                    reward_breakdown['pickup_attempt'] = pickup_spam_penalty
            else:
                pickup_far_penalty = -0.05  # Reduced penalty for exploration
                reward += pickup_far_penalty
                reward_breakdown['pickup_attempt'] = pickup_far_penalty
        else:
            reward_breakdown['pickup_attempt'] = 0.0
            
        # 9. Enhanced target visibility and proximity rewards with debugging
        try:
            objects = self.env.last_event.metadata.get("objects", [])
            target_type = self.episode_data.get("targetObjectType") if self.episode_data else None
            
            # DEBUG: Print scene objects every 10 steps to understand what's available
            if self._current_step % 10 == 0 and target_type:
                visible_objects = [obj["objectType"] for obj in objects if obj.get("visible", False)]
                all_objects = [obj["objectType"] for obj in objects]
                print(f"[DEBUG OBJECTS] Step {self._current_step}, Target: '{target_type}'")
                print(f"[DEBUG OBJECTS] Visible objects: {visible_objects[:10]}")  # Show first 10
                print(f"[DEBUG OBJECTS] All objects: {all_objects[:10]}")  # Show first 10
                
                # Check for partial matches
                target_matches = []
                for obj in objects:
                    obj_type = obj["objectType"].lower()
                    if target_type.lower() in obj_type or obj_type in target_type.lower():
                        target_matches.append(obj["objectType"])
                if target_matches:
                    print(f"[DEBUG OBJECTS] Potential target matches: {target_matches}")
            
            if target_type:
                agent_pos = self.env.last_event.metadata["agent"]["position"]
                target_visible = False
                target_visible_in_manip = False
                closest_distance = float('inf')
                
                # Try multiple matching strategies
                for obj in objects:
                    if not obj.get("visible", False):
                        continue
                        
                    obj_type = obj["objectType"]
                    is_target = False
                    
                    # Strategy 1: Exact startswith match
                    if obj_type.startswith(target_type):
                        is_target = True
                    # Strategy 2: Case-insensitive contains
                    elif target_type.lower() in obj_type.lower():
                        is_target = True
                    # Strategy 3: Reverse contains (target contains obj_type)
                    elif obj_type.lower() in target_type.lower():
                        is_target = True
                    # Strategy 4: WordNet synset matching (for SPOC compatibility)
                    elif any(word in obj_type.lower() for word in target_type.lower().split()):
                        is_target = True
                    
                    if is_target:
                        target_visible = True
                        obj_distance = math.sqrt(
                            (agent_pos["x"] - obj["position"]["x"])**2 +
                            (agent_pos["z"] - obj["position"]["z"])**2
                        )
                        closest_distance = min(closest_distance, obj_distance)
                        print(f"[DEBUG MATCH] Found target match: '{obj_type}' for target '{target_type}', distance: {obj_distance:.2f}m")
                        
                        if obj_distance < 1.5:  # Close enough to be in manipulation view
                            target_visible_in_manip = True
                
                # Big reward for seeing target object
                if target_visible:
                    visibility_reward = 2.0
                    reward += visibility_reward
                    reward_breakdown['target_visible'] = visibility_reward
                    print(f"[REWARD] 🎯 TARGET VISIBLE! Distance: {closest_distance:.2f}m, Reward: +{visibility_reward}")
                else:
                    reward_breakdown['target_visible'] = 0.0
                
                # Extra reward for being close to target
                if target_visible_in_manip:
                    manip_view_reward = 3.0  # Increased reward
                    reward += manip_view_reward
                    reward_breakdown['manipulation_view'] = manip_view_reward
                    print(f"[REWARD] 🚀 TARGET IN REACH! Reward: +{manip_view_reward}")
                else:
                    reward_breakdown['manipulation_view'] = 0.0
            else:
                reward_breakdown['target_visible'] = 0.0
                reward_breakdown['manipulation_view'] = 0.0
        except Exception as e:
            print(f"[ERROR] Error in target detection: {e}")
            reward_breakdown['target_visible'] = 0.0
            reward_breakdown['manipulation_view'] = 0.0
            
        # Update tracking variables for next step
        self.prev_distance_to_target = current_distance
        self.prev_holding = current_holding
        self.prev_position = curr_pos.copy()
        if len(action_list) > 0:
            self.action_history.append(action_list[0])
            if len(self.action_history) > 10:  # Keep only recent history
                self.action_history.pop(0)
                
        # Debug output for reward breakdown
        print(f"[DEBUG REWARD] Total: {reward:.3f}, Breakdown: {reward_breakdown}")
        
        return reward, reward_breakdown
    
    def _get_arm_state(self) -> str:
        """Get the current arm state with full SPOC-compatible proprioception."""
        try:
            # Get full arm proprioception like SPOC
            arm_proprioception = self._get_arm_proprioception()
            held_objects = self.env.last_event.metadata['arm']['heldObjects']
            
            if held_objects:
                object_name = held_objects[0]['objectType'].split('_')[0].lower()
                gripper_state = f"holding a {object_name}"
            else:
                gripper_state = "empty"
                
            # Return full state information like SPOC
            return (f"Arm at (x={arm_proprioception[0]:.2f}, y={arm_proprioception[1]:.2f}, "
                   f"z={arm_proprioception[2]:.2f}, wrist={arm_proprioception[3]:.1f}°), "
                   f"gripper is {gripper_state}.")
        
        except (KeyError, IndexError):
            return "Arm state is unavailable."
    
    def _get_arm_proprioception(self):
        """
        Get full arm proprioception like SPOC's get_arm_proprioception method.
        Returns [x, y, z, wrist_rotation] in robot coordinate frame.
        """
        try:
            arm_metadata = self.env.last_event.metadata['arm']
            joints = arm_metadata['joints']
            
            # Get wrist position (like SPOC's get_arm_wrist_position)
            wrist_joint = joints[-1]  # Last joint is the wrist
            assert wrist_joint["name"] == "stretch_robot_wrist_2_jnt"
            
            x = wrist_joint["rootRelativePosition"]["x"]
            y = wrist_joint["rootRelativePosition"]["y"]
            z = wrist_joint["rootRelativePosition"]["z"]
            
            # Get wrist rotation (like SPOC's get_arm_wrist_rotation)
            wrist_rotation = math.fmod(
                wrist_joint["rootRelativeRotation"]["w"] * wrist_joint["rootRelativeRotation"]["y"], 
                360
            )
            
            return [x, y, z, wrist_rotation]
            
        except (KeyError, IndexError, AssertionError):
            # Fallback to basic arm state
            return [0.0, 0.8, 0.0, 0.0]

    def _analyze_real_visual_scene(self, pil_image):
        """
        Generate visual description from real AI2-THOR image.
        This method analyzes the actual rendered image to describe what the robot sees.
        """
        try:
            # For now, analyze the image statistics to ensure it's real
            import numpy as np
            img_array = np.array(pil_image)
            mean_brightness = np.mean(img_array)
            std_brightness = np.std(img_array)
            
            # Determine scene type based on image characteristics
            if mean_brightness < 50:
                brightness_desc = "dimly lit"
            elif mean_brightness > 150:
                brightness_desc = "brightly lit"
            else:
                brightness_desc = "moderately lit"
            
            # Enhanced description with target information
            description = f"The robot observes a {brightness_desc} indoor household environment"
            
            # Add warning if last action failed
            if hasattr(self, '_last_action_success') and not self._last_action_success:
                description += " [WARNING: Last action failed due to collision or physical constraints]"
            
            if self.episode_data and self.episode_data.get("targetObjectType"):
                target_type = self.episode_data["targetObjectType"]
                # Get visible objects from AI2-THOR
                objects = self.env.last_event.metadata.get("objects", [])
                target_visible = False
                
                for obj in objects:
                    if obj.get("visible", False) and obj["objectType"].startswith(target_type):
                        target_visible = True
                        break
                
                if target_visible:
                    description += f". TARGET FOUND: A {target_type} is visible in the scene! The robot should approach and pick it up."
                else:
                    description += f". The robot is searching for a {target_type}. Continue exploring to find the target object."
            else:
                description += " with various objects and furniture."
                
            return description
                
        except Exception as e:
            print(f"Warning: Error analyzing real visual scene: {e}")
            return "The robot observes an indoor household environment."

    def _analyze_visual_scene(self, pil_image):
        """
        Generate visual description based on SPOC episode metadata (fallback method).
        This is used when real rendering fails.
        """
        try:
            return self._generate_metadata_based_visual_description()
        except Exception as e:
            print(f"Warning: Error generating metadata-based visual description: {e}")
            return "The robot observes an indoor household environment."
    
    def _generate_metadata_based_visual_description(self):
        """
        Generate basic visual descriptions (simplified fallback).
        This is only used if real rendering completely fails.
        """
        try:
            if self.episode_data and self.episode_data.get("targetObjectType"):
                target_type = self.episode_data["targetObjectType"]
                return f"The robot observes an indoor household environment. Currently searching for a {target_type} to complete the fetch task."
            else:
                return "The robot observes an indoor household environment with various objects and furniture."
        except Exception:
            return "The robot observes an indoor household environment."
    
    
    def _generate_synthetic_frame(self, camera_type):
        """Generate a simple synthetic frame (emergency fallback only)."""
        import numpy as np
        
        # Create simple frames as absolute fallback
        if camera_type == "navigation":
            base_color = [100, 120, 140]  # Bluish indoor environment
        else:
            base_color = [120, 100, 80]   # Warmer close-up view
        
        # Create base frame with simple texture
        frame = np.full((224, 224, 3), base_color, dtype=np.uint8)
        noise = np.random.randint(-20, 21, frame.shape)
        frame = np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)
        
        return frame

    def get_top_down_map(self, include_path=True, path_width=0.045):
        """
        Generate a SPOC-compatible top-down map with proper camera management.
        Based on SPOC's get_top_down_path_view method.
        
        Args:
            include_path: Whether to include the agent's path visualization
            path_width: Width of the path visualization
            
        Returns:
            numpy array: Top-down map image
        """
        try:
            # Store original resolution like SPOC
            original_hw = self.env.last_event.frame.shape[:2]
            map_height_width = (getattr(self.config, 'map_size', 512), getattr(self.config, 'map_size', 512))
            
            # Change resolution for map generation if needed
            if original_hw != map_height_width:
                self.env.step(
                    action="ChangeResolution", 
                    x=map_height_width[1], 
                    y=map_height_width[0], 
                    raise_for_failure=True
                )
            
            # Setup third-party camera like SPOC
            if len(self.env.last_event.third_party_camera_frames) < 2:
                event = self.env.step(action="GetMapViewCameraProperties", raise_for_failure=True)
                if event.metadata['lastActionSuccess']:
                    cam = event.metadata["actionReturn"].copy()
                    cam["orthographicSize"] += 1  # SPOC adds 1 to orthographic size
                    
                    self.env.step(
                        action="AddThirdPartyCamera",
                        skyboxColor="white",
                        **cam,
                        raise_for_failure=True
                    )
            
            # Create waypoints for target objects (SPOC-style)
            waypoints = []
            if self.episode_data and self.episode_data.get("targetObjectType"):
                target_type = self.episode_data["targetObjectType"]
                objects = self.env.last_event.metadata.get("objects", [])
                
                for obj in objects:
                    if self._is_object_type_match(obj["objectType"], target_type):
                        target_dict = {
                            "position": obj["position"],
                            "color": {"r": 1, "g": 0, "b": 0, "a": 1},  # Red for targets
                            "radius": 0.5,
                            "text": "",
                        }
                        waypoints.append(target_dict)
            
            # Add waypoints if any targets found
            if waypoints:
                self.env.step(
                    action="VisualizeWaypoints",
                    waypoints=waypoints,
                    raise_for_failure=True
                )
            
            # Add path visualization (SPOC-style)
            if include_path and hasattr(self, 'agent_path') and len(self.agent_path) > 0:
                self.env.step(
                    action="VisualizePath",
                    positions=self.agent_path,
                    pathWidth=path_width,
                    raise_for_failure=True
                )
                # Hide path immediately after capturing (like SPOC)
                self.env.step(action="HideVisualizedPath")
            
            # Get the map from third party camera
            if len(self.env.last_event.third_party_camera_frames) > 0:
                map_frame = self.env.last_event.third_party_camera_frames[-1]
                
                # Apply SPOC's cutoff (remove black borders)
                cutoff = round(map_frame.shape[1] * 6 / 396)
                if cutoff > 0:
                    map_frame = map_frame[:, cutoff:-cutoff, :]
                
                # Ensure RGB format
                if len(map_frame.shape) == 3 and map_frame.shape[2] == 4:
                    map_frame = map_frame[:, :, :3]  # Drop alpha channel
            else:
                # Fallback to main camera with cutoff
                map_frame = self.env.last_event.frame
                cutoff = round(map_frame.shape[1] * 6 / 396)
                if cutoff > 0:
                    map_frame = map_frame[:, cutoff:-cutoff, :]
            
            # Restore original resolution like SPOC
            if original_hw != map_height_width:
                self.env.step(
                    action="ChangeResolution", 
                    x=original_hw[1], 
                    y=original_hw[0], 
                    raise_for_failure=True
                )
            
            return map_frame
            
        except Exception as e:
            print(f"[ERROR] Failed to generate SPOC-compatible map: {e}")
            # Return a placeholder gray image
            return np.full((512, 512, 3), 128, dtype=np.uint8)
    
    def _pad_to_square(self, image_array):
        """
        Pad an image array to square dimensions by adding white pixels.
        
        Args:
            image_array: numpy array of shape (H, W, 3)
            
        Returns:
            Square numpy array with white padding
        """
        height, width = image_array.shape[:2]
        
        if height == width:
            return image_array  # Already square
        
        # Determine the size of the square (larger dimension)
        square_size = max(height, width)
        
        # Create white square canvas
        if len(image_array.shape) == 3:
            padded_image = np.full((square_size, square_size, image_array.shape[2]), 255, dtype=image_array.dtype)
        else:
            padded_image = np.full((square_size, square_size), 255, dtype=image_array.dtype)
        
        # Calculate padding offsets to center the original image
        y_offset = (square_size - height) // 2
        x_offset = (square_size - width) // 2
        
        # Place the original image in the center
        padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = image_array
        
        print(f"[MAP PADDING] Original: {height}×{width} → Padded: {square_size}×{square_size}")
        
        return padded_image
    
    def generate_spoc_map(self, agent_path=None, target_objects=None, map_size=(512, 512)):
        """Generate SPOC-style map with path and target visualization (ported from test_spoc_map_final.py)."""
        try:
            if agent_path is None:
                agent_path = self.agent_path
            
            print(f"[MAP DEBUG] Agent path has {len(agent_path)} points")
            if len(agent_path) > 0:
                print(f"[MAP DEBUG] First point: {agent_path[0]}")
                print(f"[MAP DEBUG] Last point: {agent_path[-1]}")
            
            # Setup third party camera if needed
            if len(self.env.last_event.third_party_camera_frames) < 2:
                event = self.env.step(action="GetMapViewCameraProperties")
                if event.metadata.get('lastActionSuccess', False):
                    cam = event.metadata["actionReturn"].copy()
                    # Increase orthographic size for better view
                    if "orthographicSize" in cam:
                        cam["orthographicSize"] += 1
                    
                    self.env.step(
                        action="AddThirdPartyCamera",
                        **cam,
                        skyboxColor="white"
                    )
            
            # Get the current map frame
            event = self.env.last_event
            if len(event.third_party_camera_frames) > 0:
                map_frame = event.third_party_camera_frames[-1].copy()
                
                # Apply cutoff like SPOC does
                cutoff = round(map_frame.shape[1] * 6 / 396)
                map_frame = map_frame[:, cutoff:-cutoff, :]
                
                # Convert RGBA to RGB if needed
                if map_frame.shape[2] == 4:
                    map_frame = map_frame[:, :, :3]
                
                # Convert to PIL for drawing
                try:
                    from PIL import Image, ImageDraw
                except ImportError:
                    print("Warning: PIL not available for map drawing")
                    return map_frame
                map_pil = Image.fromarray(map_frame.astype(np.uint8))
                draw = ImageDraw.Draw(map_pil)
                
                # Better coordinate conversion based on scene bounds
                def world_to_map(pos):
                    try:
                        # Get scene bounds for proper scaling
                        scene_bounds = self.env.last_event.metadata.get('sceneBounds')
                        if scene_bounds:
                            center = scene_bounds['center']
                            size = scene_bounds['size']
                            
                            # Convert world coordinates to map coordinates
                            rel_x = (pos['x'] - center['x']) / size['x'] * 0.8 + 0.5  # 0.8 for margin
                            rel_z = (pos['z'] - center['z']) / size['z'] * 0.8 + 0.5  # 0.8 for margin
                            
                            map_x = int(rel_x * map_pil.width)
                            map_y = int(rel_z * map_pil.height)
                        else:
                            # Fallback to automatic detection from agent path
                            if len(agent_path) > 1:
                                xs = [p['x'] for p in agent_path]
                                zs = [p['z'] for p in agent_path]
                                x_min, x_max = min(xs), max(xs)
                                z_min, z_max = min(zs), max(zs)
                                
                                # Add some margin
                                x_range = max(x_max - x_min, 5) * 1.2
                                z_range = max(z_max - z_min, 5) * 1.2
                                x_center = (x_min + x_max) / 2
                                z_center = (z_min + z_max) / 2
                            else:
                                # Very basic fallback
                                x_range = z_range = 10
                                x_center = z_center = 0
                            
                            map_x = int((pos['x'] - x_center + x_range/2) / x_range * map_pil.width)
                            map_y = int((pos['z'] - z_center + z_range/2) / z_range * map_pil.height)
                        
                        # Clamp to map bounds
                        map_x = max(0, min(map_pil.width - 1, map_x))
                        map_y = max(0, min(map_pil.height - 1, map_y))
                        
                        return map_x, map_y
                    except Exception as e:
                        print(f"Warning: Error in coordinate conversion: {e}")
                        return map_pil.width // 2, map_pil.height // 2
                
                # Draw agent path with gradient colors
                if len(agent_path) > 1:
                    path_points = [world_to_map(pos) for pos in agent_path]
                    print(f"[MAP DEBUG] Drawing path with {len(path_points)} points")
                    
                    # Draw path segments with gradient from blue (start) to red (current)
                    for i in range(len(path_points) - 1):
                        # Calculate color gradient based on position in path
                        progress = i / max(len(path_points) - 1, 1)
                        
                        # Gradient from blue (start) to red (current)
                        red = int(progress * 255)
                        blue = int((1 - progress) * 255)
                        green = 0
                        
                        # Draw thicker line for better visibility
                        draw.line([path_points[i], path_points[i+1]], 
                                fill=(red, green, blue), width=4)
                        
                        # Draw small circles at each waypoint
                        x, y = path_points[i]
                        draw.ellipse([(x-2, y-2), (x+2, y+2)], 
                                   fill=(red, green, blue), outline=(255, 255, 255))
                
                # Draw target objects
                if target_objects:
                    for obj_type in target_objects:
                        for obj in self.env.last_event.metadata.get('objects', []):
                            if obj_type.lower() in obj.get('objectType', '').lower():
                                obj_x, obj_y = world_to_map(obj['position'])
                                # Draw red circle for target
                                draw.ellipse(
                                    [(obj_x-10, obj_y-10), (obj_x+10, obj_y+10)], 
                                    fill=(255, 0, 0), 
                                    outline=(200, 0, 0)
                                )
                                break
                
                # Draw current agent position with a distinctive marker
                if agent_path:
                    agent_x, agent_y = world_to_map(agent_path[-1])
                    # Draw larger bright green circle for current position
                    draw.ellipse(
                        [(agent_x-8, agent_y-8), (agent_x+8, agent_y+8)], 
                        fill=(0, 255, 0), 
                        outline=(255, 255, 255), width=2
                    )
                    # Draw smaller inner circle for contrast
                    draw.ellipse(
                        [(agent_x-3, agent_y-3), (agent_x+3, agent_y+3)], 
                        fill=(255, 255, 255)
                    )
                    print(f"[MAP DEBUG] Agent position: ({agent_x}, {agent_y}) from world ({agent_path[-1]['x']:.1f}, {agent_path[-1]['z']:.1f})")
                    
                return np.array(map_pil)
            else:
                return np.full((224, 396, 3), 128, dtype=np.uint8)
                
        except Exception as e:
            print(f"Map generation error: {e}")
            import traceback
            traceback.print_exc()
            return np.full((224, 396, 3), 128, dtype=np.uint8)
    
    def get_multi_room_map(self, orthographic_scale=3.0, include_current_position=True):
        """
        Generate a larger-scale top-down map showing multiple rooms.
        
        Args:
            orthographic_scale: Multiplier for orthographic size (larger = more rooms visible)
            include_current_position: Whether to mark current agent position
            
        Returns:
            numpy array: Multi-room top-down map image
        """
        try:
            # Store original resolution
            original_hw = self.env.last_event.frame.shape[:2]
            map_size = getattr(self.config, 'map_size', 800)  # Larger default for multi-room
            
            # Change resolution for map generation
            self.env.step(action="ChangeResolution", x=map_size, y=map_size)
            
            # Force refresh and get current position
            current_agent_pos = self.env.last_event.metadata["agent"]["position"]
            
            # Get map view camera properties and modify for larger view
            event = self.env.step(action="GetMapViewCameraProperties")
            if event.metadata['lastActionSuccess']:
                cam = event.metadata["actionReturn"].copy()
                # Significantly increase orthographic size for multi-room view
                original_size = cam.get("orthographicSize", 10)
                cam["orthographicSize"] = original_size * orthographic_scale
                
                # Position camera higher for better overview
                if 'position' in cam:
                    cam['position']['y'] += 3  # Higher camera for multi-room view
                
                # Add third party camera for top-down view
                self.env.step(
                    action="AddThirdPartyCamera",
                    skyboxColor="white",
                    **cam
                )
            else:
                # Fallback: manually create camera with large view
                bounds = self.env.last_event.metadata.get("sceneBounds", {})
                if bounds:
                    center = bounds.get("center", {"x": 0, "y": 0, "z": 0})
                    size = bounds.get("size", {"x": 20, "y": 10, "z": 20})
                    
                    # Use much larger orthographic size for multi-room
                    ortho_size = max(size["x"], size["z"]) * orthographic_scale
                    cam_height = center["y"] + ortho_size * 0.8
                    
                    cam = {
                        "position": {"x": center["x"], "y": cam_height, "z": center["z"]},
                        "rotation": {"x": 90, "y": 0, "z": 0},
                        "orthographic": True,
                        "orthographicSize": ortho_size,
                        "fieldOfView": 90,
                        "nearClippingPlane": 0.01,
                        "farClippingPlane": 100,
                        "skyboxColor": "white"
                    }
                    
                    self.env.step(action="AddThirdPartyCamera", **cam)
            
            # Add current agent position as red waypoint
            if include_current_position:
                agent_waypoint = [{
                    "position": current_agent_pos,
                    "color": {"r": 1, "g": 0, "b": 0, "a": 1},  # Red for current position
                    "radius": 0.4,
                    "text": ""
                }]
                self.env.step(action="VisualizeWaypoints", waypoints=agent_waypoint)
            
            # Add target object highlights if available
            if self.episode_data and self.episode_data.get("targetObjectType"):
                target_type = self.episode_data["targetObjectType"]
                objects = self.env.last_event.metadata.get("objects", [])
                target_waypoints = []
                
                for obj in objects:
                    if obj["objectType"].lower().startswith(target_type.lower()):
                        target_waypoints.append({
                            "position": obj["position"],
                            "color": {"r": 0, "g": 0, "b": 1, "a": 1},  # Blue for targets
                            "radius": 0.3,
                            "text": ""
                        })
                
                if target_waypoints:
                    self.env.step(action="VisualizeWaypoints", waypoints=target_waypoints)
            
            # Visualize agent path if available
            if hasattr(self, 'agent_path') and len(self.agent_path) > 1:
                self.env.step(
                    action="VisualizePath",
                    positions=self.agent_path,
                    pathWidth=0.08,  # Thicker path for visibility
                    pathColor={"r": 1, "g": 0.5, "b": 0, "a": 1}  # Orange path
                )
            
            # Get the map from third party camera
            if len(self.env.last_event.third_party_camera_frames) > 0:
                map_frame = self.env.last_event.third_party_camera_frames[-1]
                # Convert RGBA to RGB if needed
                if len(map_frame.shape) == 3 and map_frame.shape[2] == 4:
                    map_frame = map_frame[:, :, :3]
            else:
                map_frame = self.env.last_event.frame
            
            # Clean up visualizations
            self.env.step(action="HideVisualizedPath")
            
            # Restore original resolution
            self.env.step(action="ChangeResolution", x=original_hw[1], y=original_hw[0])
            
            return map_frame
            
        except Exception as e:
            print(f"[ERROR] Failed to generate multi-room map: {e}")
            return np.full((800, 800, 3), 128, dtype=np.uint8)
    
    def _render(self, init_obs=True):
        """Render the environment observation with 3 separate images: navigation camera, manipulation camera, and top-down map."""
        img_placeholder = getattr(self.config, "image_placeholder", "<image>")
        
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False
        )
        
        # Get 3 separate images: navigation camera, manipulation camera, and map
        try:
            import numpy as np
            
            # Force a step to ensure we have a fresh event
            if not self.env.last_event or self.env.last_event.frame is None:
                print("[DEBUG] Refreshing AI2-THOR event...")
                self.env.step(action="Pass")  # Do nothing action to refresh
            
            images = []
            
            # 1. Get navigation camera
            nav_frame = self.navigation_camera
            if nav_frame is not None:
                nav_image = convert_numpy_to_PIL(nav_frame)
                images.append(nav_image)
                print(f"[SUCCESS] Got navigation camera: shape={nav_frame.shape}")
            else:
                print("[FALLBACK] Navigation camera unavailable, using synthetic")
                nav_frame_synthetic = self._generate_synthetic_frame("navigation")
                nav_image = convert_numpy_to_PIL(nav_frame_synthetic)
                images.append(nav_image)
            
            # 2. Get manipulation camera
            manip_frame = self.manipulation_camera
            if manip_frame is not None:
                manip_image = convert_numpy_to_PIL(manip_frame)
                images.append(manip_image)
                print(f"[SUCCESS] Got manipulation camera: shape={manip_frame.shape}")
            else:
                print("[FALLBACK] Manipulation camera unavailable, using synthetic")
                manip_frame_synthetic = self._generate_synthetic_frame("manipulation")
                manip_image = convert_numpy_to_PIL(manip_frame_synthetic)
                images.append(manip_image)
            
            # 3. Generate top-down map in natural scene dimensions
            try:
                target_objects = [self.episode_data.get('targetObjectType', '')] if self.episode_data else []
                # Generate map in natural dimensions (let AI2-THOR decide the aspect ratio)
                map_frame = self.generate_spoc_map(
                    agent_path=self.agent_path,
                    target_objects=target_objects,
                    map_size=None  # Let it use natural scene dimensions
                )
                
                # Agent receives the original natural dimensions
                map_image = convert_numpy_to_PIL(map_frame)
                images.append(map_image)
                print(f"[SUCCESS] Generated top-down map: natural dimensions={map_frame.shape}")
            except Exception as map_error:
                print(f"[FALLBACK] Map generation failed: {map_error}, using placeholder")
                # Use camera dimensions for fallback
                nav_h, nav_w = nav_frame.shape[:2] if nav_frame is not None else (224, 384)
                placeholder_map = np.full((nav_h, nav_w, 3), 128, dtype=np.uint8)
                map_image = convert_numpy_to_PIL(placeholder_map)
                images.append(map_image)
            
            # Generate visual description from the navigation camera (primary view)
            visual_description = self._analyze_real_visual_scene(images[0])
            
        except Exception as e:
            print(f"[CRITICAL FALLBACK] All rendering failed: {e}, using 3 synthetic images")
            
            # Complete fallback to synthetic images
            import numpy as np
            nav_frame = self._generate_synthetic_frame("navigation") 
            manip_frame = self._generate_synthetic_frame("manipulation")
            map_frame = np.full((512, 512, 3), 128, dtype=np.uint8)
            
            images = [
                convert_numpy_to_PIL(nav_frame),
                convert_numpy_to_PIL(manip_frame),
                convert_numpy_to_PIL(map_frame)
            ]
            visual_description = self._analyze_visual_scene(images[0])
        
        # For VLM, provide 3 separate images as a list
        multi_modal_data = {
            img_placeholder: images  # List of 3 PIL images: [nav_cam, manip_cam, map]
        }
        
        print(f"[RENDER SUCCESS] Provided {len(images)} images to VLM: nav_cam, manip_cam, map")
        # Note: Map is now always included as the 3rd image, no optional config needed
        
        # Get current arm state
        arm_state = self._get_arm_state()
        
        # Format the template with both visual description AND 3 <image> placeholders for VLM
        # We provide 3 images: navigation camera, manipulation camera, and top-down map
        visual_observation_text = f"{visual_description} {img_placeholder} {img_placeholder} {img_placeholder}"
        
        if init_obs:
            obs_str = init_observation_template(
                observation=f"Visual Observation: {visual_observation_text}",
                instruction=self.episode_language_instruction,
                arm_state=arm_state.replace("Arm is at", "").replace(", gripper is", ", gripper=").strip().rstrip(".")
            ) + "\n" + format_prompt_text
        else:
            obs_str = action_template(
                valid_action=self.valid_actions,
                observation=f"Visual Observation: {visual_observation_text}",
                reward=self.reward,
                done=self.measure_success()[0],
                instruction=self.episode_language_instruction,
                env_feedback=self.info["env_feedback"],
                arm_state=arm_state.replace("Arm is at", "").replace(", gripper is", ", gripper=").strip().rstrip(".")
            ) + "\n" + format_prompt_text
        
        return {
            "obs_str": obs_str,
            "multi_modal_data": multi_modal_data
        }
    
    def system_prompt(self):
        """Get the system prompt for the environment.
        
        Returns a prompt explaining the environment to the LLM agent,
        with different prompts for text and vision modes.
        
        Returns:
            System prompt string
        """
        # 为了避免 prompt 过长，这里不在 system_prompt 中附加示例
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False # in order to shorten the system prompt
        )
        
        # Provide a concise system prompt for the complex SPOC task
        spoc_system_prompt = (
            "You are a Stretch robot in a household environment. Your task is to find and fetch specific objects. "
            "STRATEGY: 1) Explore by moving forward/rotating until you see the target object. 2) When target is visible, approach it. 3) Extend arm and pickup the object. "
            "IMPORTANT: Keep responses concise. In <think> tags: observation (what you see), reasoning (what to do next), prediction (expected outcome). "
            "In <answer> tags: action name(s) only. "
            "VALID ACTIONS ONLY: moveahead, moveback, rotateright, rotateleft, rotateright_small, rotateleft_small, pickup, dropoff, move_arm_up, move_arm_down, move_arm_out, move_arm_in, wrist_open, wrist_close, move_arm_up_small, move_arm_down_small, move_arm_out_small, move_arm_in_small. "
            "DO NOT use: moveleft, moveright, or any other actions not listed above."
        )
        # spoc_system_prompt = (
        #     "You are an AI agent controlling a Stretch robot. Your ONLY task is to output actions to complete goals."
        #     "You MUST respond in the specified format. DO NOT generate any other text, conversation, or explanations."
        #     "First, think step-by-step inside the <think> tag. Analyze the observation and plan your next move."
        #     "Then, you MUST provide exactly one action from the list in the <answer> tag."
        #     "\n\n**CRITICAL: Your response MUST strictly follow this format:**"
        #     "\n<think>Your detailed reasoning here.</think><answer>action_name</answer>"
        #     "\n\n**Example:**"
        #     "\n<think>The object is to my left, so I need to rotate left to face it.</think><answer>rotateleft</answer>"
        #     "\n\nNow, begin."
        # )

        return spoc_system_prompt + '\n\n' + system_prompt(format=self.config.prompt_format) + '\n' + format_prompt_text
    
    def close(self):
        """Close the environment."""
        self.env.stop()
        
    def get_env_state(self):
        """
        Get the current state of the SPOC environment, focusing on visible objects and arm state.
        
        Returns:
            Dict: Contains target info, arm state, and visible objects.
        """
        # Get agent information
        agent_metadata = self.env.last_event.metadata["agent"]
        agent_position = agent_metadata["position"]
        
        # Get target information from the loaded episode data
        target_type = self.episode_data.get("targetObjectType", "unknown") if self.episode_data else "unknown"
        
        # Get visible objects
        objects = self.env.last_event.metadata["objects"]
        visible_objects = []
        
        for obj in objects:
            if obj.get("visible", False):
                obj_position = obj["position"]
                
                obj_distance = math.sqrt(
                    (agent_position["x"] - obj_position["x"])**2 +
                    (agent_position["z"] - obj_position["z"])**2
                )
                
                visible_objects.append({
                    "type": obj["objectType"],
                    "distance_to_player": round(obj_distance, 2),
                })
    
        # Sort objects by distance
        visible_objects.sort(key=lambda x: x["distance_to_player"])
        
        return {
            "target_obj_type": target_type, 
            "arm_state": self._get_arm_state(),
            "visible_objects": visible_objects[:self.config.max_objects_in_state],   
        }

if __name__ == "__main__":    # remember to "export SPOC_DATA_PATH=/home/jiajunliu/spoc_data/fifteen"    first, or on the remote server, you can use "export SPOC_DATA_PATH=root/spoc_data/fifteen"
    
    SPOC_DATA_PATH = os.environ.get("SPOC_DATA_PATH") 
    if not SPOC_DATA_PATH:
        raise ValueError("Please set the SPOC_DATA_PATH environment variable to your dataset path.")

    config = SpocEnvConfig(
        data_path=SPOC_DATA_PATH,
        task_type="FetchType",
        chores_split="train"
    )
    
    env = SpocEnv(config)
    print(env.system_prompt())
    
    obs, info = env.reset(seed=42)
    print(obs["obs_str"])
    i = 0
    os.makedirs("./test_spoc", exist_ok=True)
    
    # Save the first image
    if obs["multi_modal_data"][config.image_placeholder]:
        img = obs["multi_modal_data"][config.image_placeholder][0]
        img.save(f"./test_spoc/spoc_{i}.png")
        
    done = False
    
    # Interactive testing loop
    while not done:
        i += 1
        action = input(f"Enter action index (1-20) or name: ")
        
        # Allow entering index directly
        if action.isdigit() and int(action) in range(1, 21):
            action_name = list(env.ACTION_LOOKUP.keys())[int(action)-1]
        else:
            action_name = action.lower()

        if action_name not in env.ACTION_LOOKUP:
            print(f"Invalid action: '{action_name}'. Please choose from the list.")
            continue
            
        action_str = f"<think>User chose {action_name}. I will execute it.</think><answer>{action_name}</answer>"
        obs, reward, done, info = env.step(action_str)
        
        print("-" * 20)
        print(f"Step {i}: Executed '{action_name}'")
        print(f"Reward: {reward}, Done: {done}, Last Action Success: {info.get('last_action_success')}")
        print(obs["obs_str"])

        if obs["multi_modal_data"][config.image_placeholder]:
            img = obs["multi_modal_data"][config.image_placeholder][0]
            img.save(f"./test_spoc/spoc_{i}.png")
        
        print(f"Success Metric: {info.get('task_success')}")
        
        if done or info.get('task_success'):
            print("Episode finished.")
            break
            
        if i > env._max_episode_steps:
            print("Max steps reached.")
            break
    
    env.close()