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
            'PYOPENGL_PLATFORM': 'egl',
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
            "quality": "Low",
            # SPOC-specific parameters
            "useMassThreshold": True,
            "massThreshold": 10,
            "autoSimulation": False,
            "autoSyncTransforms": True,
            "snapToGrid": False,
            "fastActionEmit": True,
            "cameraNearPlane": 0.01,
        }

        self.env = None
        # Try different platform configurations
        platforms_to_try = []
        
        # Check if we're in a headless environment
        if not os.environ.get('DISPLAY'):
            # Headless mode - try CloudRendering first
            platforms_to_try.extend(["CloudRendering", "Linux64"])
        else:
            # Display available - try Linux64 first
            platforms_to_try.extend(["Linux64", "CloudRendering"])

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
        
        return self._render(init_obs=True), {}
    
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
                    
                    # Update reward based on success
                    if success:
                        self.reward += 10.0  # Success reward
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
        
        if metrics['turn_metrics']['action_is_valid'] and rst.get("format_correct", True):
            self.reward += self.config.format_reward
            info["is_format_rewarded"] = True
        else:
            info["is_format_rewarded"] = False
            
        # Check if the agent position has changed (action was effective)
        curr_pos = self.env.last_event.metadata["agent"]["position"]
        metrics['turn_metrics']['action_is_effective'] = curr_pos["x"] != prev_pos["x"] or curr_pos["z"] != prev_pos["z"]
        
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
            11: {"action": "MoveArmRelative", "y": 0.1},
            12: {"action": "MoveArmRelative", "y": -0.1},
            13: {"action": "MoveArmRelative", "z": 0.1},
            14: {"action": "MoveArmRelative", "z": -0.1},
            15: {"action": "RotateWristRelative", "yaw": -10},
            16: {"action": "RotateWristRelative", "yaw": 10},
            17: {"action": "MoveArmRelative", "y": 0.02},
            18: {"action": "MoveArmRelative", "y": -0.02},
            19: {"action": "MoveArmRelative", "z": 0.02},
            20: {"action": "MoveArmRelative", "z": -0.02},
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

    def _render(self, init_obs=True):
        """Render the environment observation, including the REAL image."""
        img_placeholder = getattr(self.config, "image_placeholder", "<image>")
        
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False
        )
        
        # --- Get the frame from AI2-THOR ---
        frame = self.env.last_event.frame
        if frame is None:
            # If frame is None, execute a Pass action to get the first frame
            self.env.step("Pass")
            frame = self.env.last_event.frame
            
        if frame is None:
            # If still None, create a black placeholder image
            import numpy as np
            frame = np.zeros((self.config.resolution, self.config.resolution, 3), dtype=np.uint8)
            
        multi_modal_data = {
            img_placeholder: [convert_numpy_to_PIL(frame)]
        }
        
        # Get current arm state
        arm_state = self._get_arm_state()
        
        # Format the template
        if init_obs:
            obs_str = init_observation_template(
                observation=f"Proprioception: {arm_state}\nVisuals: {img_placeholder}",
                instruction=self.episode_language_instruction,
            ) + "\n" + format_prompt_text
        else:
            obs_str = action_template(
                valid_action=self.valid_actions,
                observation=f"Proprioception: {arm_state}\nVisuals: {img_placeholder}",
                reward=self.reward,
                done=self.measure_success()[0],
                instruction=self.episode_language_instruction,
                env_feedback=self.info["env_feedback"]
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