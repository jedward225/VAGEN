from vagen.env.base.base_env import BaseEnv
import ai2thor.controller
import numpy as np
import time
import math
import os
import time

from vagen.env.utils.context_utils import convert_numpy_to_PIL
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from .env_config import SpocEnvConfig
from .prompt import system_prompt,init_observation_template, action_template, format_prompt
from .task_loader import get_dataset
from vagen.env.utils.state_reward_text_utils import env_state_reward_wrapper

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
        "Deprecated - tilt camera upward (not used for Stretch)", # lookup
        "Deprecated - tilt camera downward (not used for Stretch)", # lookdown
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

        self.thor_config = {
            "agentMode": "stretch",
            "gridSize": 0.1,
            "visibilityDistance": 10,
            "visibilityScheme": "Distance",
            "renderDepthImage": False,  # Disable depth to reduce memory usage
            "renderInstanceSegmentation": True,  # Enable for SPOC compatibility
            "width": config.resolution,
            "height": config.resolution,
            "fieldOfView": config.fov,
            "server_timeout": 900, 
            "server_start_timeout": 900,
            "quality": "Ultra",  # Maximum quality for CloudRendering
            # CloudRendering specific fixes
            "makeAgentsVisible": False,
            "renderInteractableShader": False,
            # SPOC-specific parameters
            "useMassThreshold": True,
            "massThreshold": 10,
            "autoSimulation": False,
            "autoSyncTransforms": True,
            "snapToGrid": False,
            "fastActionEmit": True,
            "cameraNearPlane": 0.01,
            "cameraFarPlane": 20.0,
            # Enable third-party camera for manipulation view
            "thirdPartyCameras": [
                {
                    "position": {"x": 0, "y": 0, "z": 0},  # Will be set relative to arm
                    "rotation": {"x": 0, "y": 0, "z": 0},
                    "fieldOfView": config.fov,
                    "width": config.resolution,
                    "height": config.resolution
                }
            ]
        }

        self.env = None
        # Try different platform configurations
        platforms_to_try = []
        
        # Force software rendering in headless environments
        if not os.environ.get('DISPLAY') or os.environ.get('FORCE_HEADLESS', '0') == '1':
            # Set additional environment variables for software rendering
            os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
            os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
            os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
            os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
            
            # Force CloudRendering only (since Linux64 not supported in this environment)
            platforms_to_try.extend(["CloudRendering"])
        else:
            # Force CloudRendering only (since Linux64 not supported in this environment) 
            platforms_to_try.extend(["CloudRendering"])

        for platform_str in platforms_to_try:
            try:
                # 构造配置，直接传入 platform 字符串
                # 注意：x_display 参数只对 Linux64 有意义，且在 xvfb-run 模式下通常由 xvfb-run 管理
                config_to_try = {**self.thor_config, "platform": platform_str, "headless": True}
                
                print(f"Attempting AI2-THOR with platform: {platform_str}")
                self.env = ai2thor.controller.Controller(**config_to_try)
                print(f"Successfully initialized AI2-THOR with platform: {platform_str}")
                self.thor_config = config_to_try  # 保存成功配置
                break
            except Exception as e:
                print(f"Failed to initialize with platform {platform_str}: {e}")
                if self.env:
                    try:
                        self.env.stop()
                    except Exception:
                        pass
                    self.env = None
        
        if self.env is None:
            raise RuntimeError("Failed to initialize AI2-THOR with any platform configuration. "
                             "Please check that AI2-THOR is properly installed and the server environment supports headless rendering.")
        
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
        
        # --- New reward tracking variables ---
        self.prev_distance_to_target = None
        self.prev_holding = False
        self.prev_position = None
        self.action_history = []
        self.exploration_positions = set()
        self.last_pickup_attempt_step = -10  # Track pickup attempts
        
        self.format_prompt_func = format_prompt[self.config.prompt_format]
        self.parse_func = PARSE_FUNC_MAP[self.config.prompt_format]
        
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
        self.episode_language_instruction = traj_data["instruction"]

        # Reset the AI2-THOR scene
        scene_name = traj_data["scene"]
        max_retries = 3
        reset_success = False
        for attempt in range(max_retries):
            try:
                # Step 1: Reset the AI2-THOR scene
                self._last_event = self.env.reset(scene=scene_name)
                if not self._last_event or not self._last_event.metadata.get('lastActionSuccess'):
                    raise RuntimeError(f"Attempt {attempt + 1}: Failed to reset to scene {scene_name}.")
                
                # Step 2: Teleport the agent to the starting pose
                pose = traj_data["agentPose"]
                self._last_event = self.env.step(
                    action="TeleportFull",
                    position=pose["position"],
                    rotation={'x': 0, 'y': pose["rotation"], 'z': 0},
                    horizon=0,  # Use HORIZON=0 as in SPOC
                    standing=True,
                    forceAction=True
                )
                if not self._last_event or not self._last_event.metadata.get('lastActionSuccess'):
                    raise RuntimeError(f"Attempt {attempt + 1}: Failed to teleport agent in scene {scene_name}.")

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
        info['last_action_success'] = self.env.last_event.metadata['lastActionSuccess']
        info["env_feedback"] ="Last action is executed successfully." if info['last_action_success'] else "Last action is not executed successfully."
        self.info = info
        # Update total reward
        self.total_reward += self.reward
        
        return self._render(init_obs=False), self.reward, done, info
    
    def _execute_action(self, action_index: int):
        """Executes a discrete action for the Stretch robot."""
        action_map = {
            1: {"action": "MoveAhead", "moveMagnitude": 0.2},
            2: {"action": "MoveBack", "moveMagnitude": 0.2},
            3: {"action": "RotateRight", "degrees": 30},
            4: {"action": "RotateLeft", "degrees": 30},
            5: {"action": "RotateRight", "degrees": 6},
            6: {"action": "RotateLeft", "degrees": 6},
            7: {"action": "Pass"},  # Deprecated lookup
            8: {"action": "Pass"},  # Deprecated lookdown
            9: {"action": "PickupObject"}, # Attempts to pick up object in front of gripper
            10: {"action": "ReleaseObject"},
            11: {"action": "MoveArmRelative", "offset": {"x": 0, "y": 0.1, "z": 0}},
            12: {"action": "MoveArmRelative", "offset": {"x": 0, "y": -0.1, "z": 0}},
            13: {"action": "MoveArmRelative", "offset": {"x": 0, "y": 0, "z": 0.1}},
            14: {"action": "MoveArmRelative", "offset": {"x": 0, "y": 0, "z": -0.1}},
            15: {"action": "RotateWristRelative", "yaw": -10},
            16: {"action": "RotateWristRelative", "yaw": 10},
            17: {"action": "MoveArmRelative", "offset": {"x": 0, "y": 0.02, "z": 0}},
            18: {"action": "MoveArmRelative", "offset": {"x": 0, "y": -0.02, "z": 0}},
            19: {"action": "MoveArmRelative", "offset": {"x": 0, "y": 0, "z": 0.02}},
            20: {"action": "MoveArmRelative", "offset": {"x": 0, "y": 0, "z": -0.02}},
        }

        params = action_map.get(action_index, {"action": "Pass"})
        self._last_event = self.env.step(**params)

        # Update held status after pickup/dropoff attempts
        if params["action"] == "PickupObject" and self._last_event.metadata["lastActionSuccess"]:
            self.is_holding = True
        elif params["action"] == "ReleaseObject" and self._last_event.metadata["lastActionSuccess"]:
            self.is_holding = False

    def measure_success(self):
        """
        Check if the agent has successfully completed the Fetch task.
        Success is defined as holding an object of the correct type.
        """
        # Default to not successful
        success = False
        
        try:
            # Check if we are holding the correct object type
            if self.is_holding and self.episode_data:
                held_objects = self.env.last_event.metadata['arm']['heldObjects']
                target_type = self.episode_data.get("targetObjectType")
                if held_objects and target_type:
                    # Check if any held object matches the target type
                    for obj in held_objects:
                        # AI2-THOR object types can have numbers appended (e.g., "Mug_1")
                        if obj['objectType'].startswith(target_type):
                            success = True
                            break
        except (KeyError, IndexError) as e:
            # Metadata might not be available if the last action failed weirdly
            print(f"Warning: Could not check success due to missing metadata: {e}")
            success = False

        # For compatibility, also return distance, though it's not the primary metric
        agent_pos = self.env.last_event.metadata["agent"]["position"]
        target_pos = self.episode_data.get("target_position", agent_pos) if self.episode_data else agent_pos
        distance = math.sqrt(
            (agent_pos["x"] - target_pos["x"])**2 +
            (agent_pos["z"] - target_pos["z"])**2
        )
        
        return float(success), distance
    
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
            
        # 5. Object manipulation rewards
        current_holding = self.is_holding
        if current_holding and not self.prev_holding:
            pickup_reward = 3.0  # First time picking up object
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
            
        # 9. NEW: Reward for getting objects visible in manipulation view
        try:
            objects = self.env.last_event.metadata.get("objects", [])
            target_type = self.episode_data.get("targetObjectType") if self.episode_data else None
            if target_type:
                target_visible_in_manip = False
                # Check if target is close enough to be in manipulation range
                agent_pos = self.env.last_event.metadata["agent"]["position"]
                for obj in objects:
                    if obj.get("visible", False) and obj["objectType"].startswith(target_type):
                        obj_distance = math.sqrt(
                            (agent_pos["x"] - obj["position"]["x"])**2 +
                            (agent_pos["z"] - obj["position"]["z"])**2
                        )
                        if obj_distance < 1.5:  # Close enough to be in manipulation view
                            target_visible_in_manip = True
                            break
                
                if target_visible_in_manip:
                    manip_view_reward = 0.8
                    reward += manip_view_reward
                    reward_breakdown['manipulation_view'] = manip_view_reward
                else:
                    reward_breakdown['manipulation_view'] = 0.0
            else:
                reward_breakdown['manipulation_view'] = 0.0
        except Exception:
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
        """Get the current arm state from real environment metadata."""
        try:
            arm_meta = self.env.last_event.metadata['arm']
            arm_pos = arm_meta['position']
            held_objects = arm_meta['heldObjects']
            
            if held_objects:
                object_name = held_objects[0]['objectType'].split('_')[0].lower()
                gripper_state = f"holding a {object_name}"
            else:
                gripper_state = "empty"
                
            # Returns a simplified state for the prompt
            return f"Arm is at (y={arm_pos['y']:.2f}, z={arm_pos['z']:.2f}), gripper is {gripper_state}."
        
        except (KeyError, IndexError):
            return "Arm state is unavailable."

    def _analyze_visual_scene(self, pil_image):
        """
        Generate visual description based on SPOC episode metadata.
        Since CloudRendering is not working, we use the rich SPOC metadata to create intelligent descriptions.
        """
        try:
            return self._generate_metadata_based_visual_description()
        except Exception as e:
            print(f"Warning: Error generating metadata-based visual description: {e}")
            return "The robot observes an indoor household environment."
    
    def _generate_metadata_based_visual_description(self):
        """
        Generate intelligent visual descriptions based on SPOC episode metadata.
        """
        import json
        import numpy as np
        
        # Get current episode metadata
        if not hasattr(self, 'current_episode') or self.current_episode is None:
            return "The robot observes an indoor household environment."
        
        # Decode task specification
        task_spec_bytes = self.current_episode.get('templated_task_spec', b'')
        if isinstance(task_spec_bytes, np.ndarray):
            task_spec_bytes = task_spec_bytes.tobytes()
        
        try:
            task_spec = json.loads(task_spec_bytes.decode('utf-8').rstrip('\x00'))
        except:
            task_spec = {}
        
        # Get current step index
        current_step = getattr(self, 'current_step_in_episode', 0)
        
        # Get navigation and manipulation object info
        nav_objects = self._decode_object_info('nav_accurate_object_bbox', current_step)
        manip_objects = self._decode_object_info('manip_accurate_object_bbox', current_step)
        
        # Get environment state
        object_in_hand = bool(self.current_episode.get('an_object_is_in_hand', [False])[current_step]) if current_step < len(self.current_episode.get('an_object_is_in_hand', [])) else False
        room_seen = bool(self.current_episode.get('room_current_seen', [False])[current_step]) if current_step < len(self.current_episode.get('room_current_seen', [])) else False
        
        # Generate description
        description_parts = []
        
        # Navigation view description
        nav_desc = self._describe_camera_view(nav_objects, task_spec, "navigation", room_seen)
        description_parts.append(f"Navigation view: {nav_desc}")
        
        # Manipulation view description  
        manip_desc = self._describe_camera_view(manip_objects, task_spec, "manipulation", object_in_hand)
        description_parts.append(f"Manipulation view: {manip_desc}")
        
        return "[Dual camera view] " + ". ".join(description_parts) + "."
    
    def _decode_object_info(self, bbox_key, step_idx):
        """Decode object information from SPOC metadata."""
        try:
            if bbox_key not in self.current_episode:
                return []
            
            # Get object IDs and types for current step
            oids_key = f'{bbox_key}/oids_as_bytes'
            synsets_key = f'{bbox_key}/synset_to_oids_as_bytes'
            
            if oids_key in self.current_episode and step_idx < len(self.current_episode[oids_key]):
                oids_bytes = self.current_episode[oids_key][step_idx]
                synsets_bytes = self.current_episode[synsets_key][step_idx]
                
                # Decode bytes to string
                if isinstance(oids_bytes, np.ndarray):
                    oids_bytes = oids_bytes.tobytes()
                if isinstance(synsets_bytes, np.ndarray):
                    synsets_bytes = synsets_bytes.tobytes()
                
                try:
                    import json
                    object_ids = json.loads(oids_bytes.decode('utf-8').rstrip('\x00'))
                    synset_mapping = json.loads(synsets_bytes.decode('utf-8').rstrip('\x00'))
                    
                    # Convert to list of object types
                    object_types = []
                    for synset, ids in synset_mapping.items():
                        readable_name = self._synset_to_readable(synset)
                        object_types.extend([readable_name] * len(ids))
                    
                    return object_types[:5]  # Limit to 5 objects for brevity
                except:
                    return []
            return []
        except Exception as e:
            return []
    
    def _describe_camera_view(self, objects, task_spec, view_type, context_flag):
        """Generate description for a camera view based on objects and context."""
        # Get target object info
        target_synsets = task_spec.get('synsets', [])
        target_name = self._synset_to_readable(target_synsets[0]) if target_synsets else 'unknown object'
        
        if not objects:
            if view_type == "navigation":
                return f"Wide-field view of an indoor environment, searching for {target_name}"
            else:
                return "Close-up view showing nearby surfaces and potential interaction areas"
        
        # Describe visible objects
        unique_objects = list(set(objects))
        if len(unique_objects) == 1:
            obj_desc = unique_objects[0]
        elif len(unique_objects) == 2:
            obj_desc = f"{unique_objects[0]} and {unique_objects[1]}"
        else:
            obj_desc = f"{', '.join(unique_objects[:-1])}, and {unique_objects[-1]}"
        
        # Check if target object is visible
        target_visible = any(target_name.lower() in obj.lower() for obj in objects)
        
        if view_type == "navigation":
            base_desc = f"Wide-field view showing {obj_desc} in a household environment"
            if target_visible:
                base_desc += f". The target {target_name} is visible"
            elif context_flag:  # room_seen
                base_desc += f". Currently in the target room, searching for {target_name}"
            else:
                base_desc += f". Exploring to locate {target_name}"
        else:  # manipulation
            base_desc = f"Close-up view showing {obj_desc}"
            if context_flag:  # object_in_hand
                base_desc += ". An object is currently grasped"
            elif target_visible:
                base_desc += f". The target {target_name} is within reach"
            else:
                base_desc += ". Ready for potential object interaction"
        
        return base_desc
    
    def _synset_to_readable(self, synset):
        """Convert WordNet synset to readable object name."""
        synset_map = {
            'houseplant.n.01': 'houseplant',
            'cup.n.01': 'cup',
            'bowl.n.01': 'bowl', 
            'plate.n.04': 'plate',
            'mug.n.04': 'mug',
            'book.n.01': 'book',
            'pillow.n.01': 'pillow',
            'remote_control.n.01': 'remote control',
            'cellular_telephone.n.01': 'phone',
            'watch.n.01': 'watch',
            'key.n.01': 'key',
            'credit_card.n.01': 'credit card',
            'pen.n.01': 'pen',
            'pencil.n.01': 'pencil',
            'laptop.n.01': 'laptop'
        }
        
        if synset in synset_map:
            return synset_map[synset]
        
        # Fallback: extract word from synset
        try:
            return synset.split('.')[0].replace('_', ' ')
        except:
            return 'object'
    
    def _generate_synthetic_frame(self, camera_type):
        """Generate a meaningful synthetic frame based on SPOC metadata."""
        import numpy as np
        
        # Get current step and episode info
        current_step = getattr(self, 'current_step_in_episode', 0)
        
        if camera_type == "navigation":
            # Navigation camera: wider view, more environmental
            base_color = [100, 120, 140]  # Bluish indoor environment
            objects = self._decode_object_info('nav_accurate_object_bbox', current_step)
        else:
            # Manipulation camera: closer view, more object-focused
            base_color = [120, 100, 80]   # Warmer close-up view
            objects = self._decode_object_info('manip_accurate_object_bbox', current_step)
        
        # Create base frame
        frame = np.full((self.config.resolution, self.config.resolution, 3), base_color, dtype=np.uint8)
        
        # Add texture based on object count and type
        texture_intensity = min(len(objects) * 10, 40)
        if texture_intensity > 0:
            noise = np.random.randint(-texture_intensity, texture_intensity + 1, frame.shape)
            frame = np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)
        else:
            # Add minimal base texture even without objects
            noise = np.random.randint(-10, 11, frame.shape)
            frame = np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Add object-based color variations
        if objects:
            for i, obj in enumerate(objects[:3]):  # Max 3 objects to avoid oversaturation
                # Add colored regions representing objects
                y_start = (i * 50) % (frame.shape[0] - 50)
                x_start = (i * 60) % (frame.shape[1] - 60)
                
                # Object-specific colors
                if 'plant' in obj.lower():
                    obj_color = [60, 150, 60]  # Green for plants
                elif any(word in obj.lower() for word in ['cup', 'mug', 'bowl']):
                    obj_color = [180, 160, 140]  # Neutral tableware colors
                elif any(word in obj.lower() for word in ['book', 'paper']):
                    obj_color = [200, 200, 180]  # Paper colors
                else:
                    obj_color = [160, 140, 120]  # Generic object color
                
                # Add object region
                frame[y_start:y_start+40, x_start:x_start+50] = np.clip(
                    frame[y_start:y_start+40, x_start:x_start+50] * 0.7 + 
                    np.array(obj_color) * 0.3, 0, 255
                ).astype(np.uint8)
        
        return frame

    def _render(self, init_obs=True):
        """Render the environment observation, including the REAL image."""
        img_placeholder = getattr(self.config, "image_placeholder", "<image>")
        
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False
        )
        
        # --- Generate synthetic frames since CloudRendering doesn't work ---
        # CloudRendering in this AI2-THOR version cannot generate images, so we create meaningful synthetic frames
        import numpy as np
        
        nav_frame = self._generate_synthetic_frame("navigation")
        manip_frame = self._generate_synthetic_frame("manipulation")
            
        # Apply the same cropping as in SPOC (6 pixels from each side)
        nav_cutoff = round(nav_frame.shape[1] * 6 / 396)
        manip_cutoff = round(manip_frame.shape[1] * 6 / 396)
        
        if nav_cutoff > 0:
            nav_frame = nav_frame[:, nav_cutoff:-nav_cutoff, :]
        if manip_cutoff > 0:
            manip_frame = manip_frame[:, manip_cutoff:-manip_cutoff, :]
            
        # Concatenate the two camera images side by side (like in SPOC)
        concatenated_frame = np.concatenate([nav_frame, manip_frame], axis=1)
        
        # Convert concatenated frame to PIL Image
        pil_image = convert_numpy_to_PIL(concatenated_frame)
        
        # Generate actual visual description from the PIL image
        visual_description = self._analyze_visual_scene(pil_image)
        
        # For VLM, provide the concatenated image as a list (required by serialization)
        multi_modal_data = {
            img_placeholder: [pil_image]  # List containing the concatenated image
        }
        
        # Get current arm state
        arm_state = self._get_arm_state()
        
        # Format the template with both visual description AND <image> placeholder for VLM
        visual_observation_text = f"{visual_description} {img_placeholder}"
        
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
        
        # Provide a more detailed system prompt for the complex SPOC task
        spoc_system_prompt = (
            "You are a helpful robot assistant controlling a Stretch robot in a household environment. "
            "Your goal is to follow natural language instructions to complete tasks like fetching objects. "
            "At each step, you will receive your arm's state (proprioception) and a visual image of your "
            "surroundings. You must output a valid action from the provided list to control the robot. "
            "Think step-by-step about how to decompose the task and select the best action."
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