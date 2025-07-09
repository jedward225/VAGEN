from vagen.env.base.base_env import BaseEnv
import ai2thor.controller
import numpy as np
import time
import math
# CloudRendering 平台需要官方服务器账号，大多数离线服务器无法使用。
# 这里仅依赖 ai2thor 标准模块即可，无需显式导入 CloudRendering。
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
        
        # --- AI2-THOR Controller Configuration ---
        import os
        # Use EGL headless rendering if no display is available (e.g., on a server)
        platform_choice = "EGL" if os.environ.get('DISPLAY') is None else "Linux64"

        self.thor_config = {
            "agentMode": "stretch",
            "renderInstanceSegmentation": True,
            "renderDepthImage": True,
            "width": config.resolution,
            "height": config.resolution,
            "fieldOfView": config.fov,
            "platform": platform_choice,
            "headless": platform_choice == "EGL",
            "gpu_device": config.get('gpu_device', 0),
            "server_timeout": 600, # Increased timeout for complex scenes
            "server_start_timeout": 600,
            "quality": "Low",
            "gridSize": 0.1,
            "visibilityDistance": 10,
        }
        
        self.env = ai2thor.controller.Controller(**self.thor_config)
        
        # --- Dataset Loading ---
        self.dataset = get_dataset(
            data_path=config.data_path, 
            task_type=config.task_type, 
            split=config.chores_split
        )
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
        try:
            self._last_event = self.env.reset(scene=scene_name)
        except Exception as e:
            print(f"Error resetting to scene {scene_name}: {e}. Trying again...")
            self.env.stop() # Force stop and restart controller
            self.env = ai2thor.controller.Controller(**self.thor_config)
            self._last_event = self.env.reset(scene=scene_name)


        # Teleport the agent to the starting position
        pose = traj_data["agentPose"]
        self._last_event = self.env.step(
            action="Teleport",
            position=pose["position"],
            rotation={'x': 0, 'y': pose["rotation"], 'z': 0},
            horizon=pose["horizon"],
            standing=True
        )

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
        rst = self.parse_func(
            response=action_str,
            special_token_list=self.config.get('special_token_list', None),
            action_sep=self.config.get('action_sep', ','),
            max_actions=self.config.get('max_actions_per_step', 1)
        )
        
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
        if metrics["turn_metrics"]["action_is_valid"] and rst.get("format_correct", True):
            
            for action in action_list:
                action_lower = action.lower()
                if action_lower in self.ACTION_LOOKUP:
                    action_int = self.ACTION_LOOKUP[action_lower]
                    self._execute_action(action_int)
                    success, distance = self.measure_success()
                    
                    # Update reward based on success
                    if success:
                        self.reward += 10.0  # Success reward
                        done = True
                        metrics['traj_metrics']['success'] = True
                    
                    self.valid_actions.append(action)
                    
                    if done:
                        break
                else:
                    metrics['turn_metrics']['action_is_valid'] = False
                    break
                
                self._current_step += 1
                if self._current_step >= self._max_episode_steps:
                    done = True
                    break
        
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
            if self.is_holding:
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
        target_pos = self.episode_data.get("target_position", agent_pos) # Use agent pos if no target
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
        img_placeholder = self.config.get("image_placeholder", "<image>")
        
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False
        )
        
        # --- FIX: Provide the actual image frame to the model ---
        frame = self.env.last_event.frame
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
        target_type = self.episode_data.get("targetObjectType", "unknown")
        
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


if __name__ == "__main__":
    # Example usage for debugging the environment
    import os
    
    # IMPORTANT: Set this to your local SPOC dataset path
    # Example: "/home/user/spoc_data/fifteen_type"
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