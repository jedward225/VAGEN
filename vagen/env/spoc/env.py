from vagen.env.base.base_env import BaseEnv
import ai2thor.controller
import numpy as np
import time
import math
from ai2thor.platform import CloudRendering
from vagen.env.utils.context_utils import convert_numpy_to_PIL
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from .env_config import SpocEnvConfig
from .prompt import system_prompt,init_observation_template, action_template, format_prompt
from .task_loader import load_chores_episode, get_episode_count
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
        """Initialize the SPOC environment.
        
        Args:
            config: Configuration for the environment including resolution, FOV,
                   eval set, render mode, etc.
        """
        super().__init__()
        self.config = config
        self.success_threshold = self.config.success_threshold
        self.step_length = self.config.step_length
        # Environment setup
        self.resolution = config.resolution
        # 检测是否有可用的显示环境
        import os
        display_available = os.environ.get('DISPLAY') is not None
        
        self.thor_config = {
            "agentMode": "default",
            "gridSize": 0.1,
            "visibilityDistance": 10,
            "renderDepthImage": False,
            "renderInstanceSegmentation": False,
            "width": self.resolution,
            "height": self.resolution,
            "fieldOfView": config.fov,
            "platform": "CloudRendering" if not display_available else "Linux64",
            "headless": False,
            "gpu_device": config.get('gpu_device', 0),
            "server_timeout": 300,
            "server_start_timeout": 300,
            "quality": "Low",   # 环境暂时设置的quality很低
        }
        
        # Initialize AI2-THOR controller with Stretch configuration
        self.thor_config.update({
            "agentMode": "stretch",
            "renderInstanceSegmentation": True,
            "renderDepthImage": True,
        })
        self.env = ai2thor.controller.Controller(**self.thor_config)
        
        # Load dataset count (use ChoresDataset instead of navigation dataset)
        self.number_of_episodes = get_episode_count(config.chores_split)
        self._current_episode_num = 0
        self._current_step = 0
        self._max_episode_steps = 30
        self._episode_start_time = 0
        self.is_holding = False
        self.episode_log = []
        self.episode_language_instruction = ""
        self.episode_data = None
        self._last_event = None
        self.standing = True
        self.multiview = config.multiview
        self.img_paths = []
        self.total_reward = 0
        self.valid_actions = []
        self.reward = 0
        
        # Store the format prompt function for later use
        self.format_prompt_func = format_prompt[self.config.prompt_format]
        
        # Get the parse function based on the prompt format
        self.parse_func = PARSE_FUNC_MAP[self.config.prompt_format]
        
    def _get_episode_count(self):
        """Get the total number of episodes available."""
        return get_episode_count(self.config.chores_split)
    
    def reset(self, seed=None):
        """Reset the environment to a new episode.
        
        This method resets the AI2-THOR environment and initializes a new episode
        based on the dataset. If a seed is provided, it ensures deterministic
        episode selection.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Observation dict, info dict
        """
        # Reset the environment with the proper seed
        
        idx = seed % self.number_of_episodes if seed is not None else np.random.randint(0, self.number_of_episodes) # in order to make sure the seed is not None
        
        # Get the trajectory data from ChoresDataset
        traj_data = load_chores_episode(self.config.chores_split, self.config.task_type, idx)
        self.episode_data = traj_data
        self.episode_language_instruction = traj_data["instruction"]

        # Reset the AI2-THOR environment
        scene_name = traj_data["scene"]
        self._last_event = self.env.reset(scene=scene_name)

        # Set up the camera for multiview if enabled
        if self.multiview:
            event = self.env.step(action="GetMapViewCameraProperties", raise_for_failure=True)
            pose = event.metadata["actionReturn"].copy()
            pose["orthographic"] = True
            self.env.step(
                action="AddThirdPartyCamera",
                **pose,
                skyboxColor="white",
                raise_for_failure=True,
            )

        # Teleport the agent to the starting position
        pose = traj_data["agentPose"]
        self.env.step(
            action="Teleport",
            position={
                "x": pose["position"]["x"],
                "y": pose["position"]["y"],
                "z": pose["position"]["z"]
            },
            rotation={
                "x": 0,
                "y": pose["rotation"],
                "z": 0
            },
            horizon=pose["horizon"],
            standing=True
        )

        # Reset episode tracking information
        self._current_step = 0
        self.standing = True
        self.episode_log = []
        self._episode_start_time = time.time()
        self.img_paths = []
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
    
    def _execute_action(self, action_index):
        """Execute the discrete action in the environment using Stretch robot APIs.
        
        Args:
            action_index: Index of the action to execute (1-20)
        """
        # Get current arm position for arm movement actions
        if action_index >= 11 and action_index <= 20:
            # For arm actions, we need to get current position and calculate new position
            try:
                current_arm = self.env.last_event.metadata.get("arm", {})
                # This is a simplified approach - real implementation would get current arm state
                # For now, use relative movements compatible with AI2-THOR
                pass
            except:
                pass
        
        if action_index == 1:  # moveahead - Move forward by 0.2 meter  
            self._last_event = self.env.step(action="MoveAgent", ahead=0.2)
        elif action_index == 2:  # moveback - Move backward by 0.2 meter
            self._last_event = self.env.step(action="MoveAgent", ahead=-0.2)
        elif action_index == 3:  # rotateright - Rotate right by 30 degrees
            self._last_event = self.env.step(action="RotateAgent", degrees=30)
        elif action_index == 4:  # rotateleft - Rotate left by 30 degrees
            self._last_event = self.env.step(action="RotateAgent", degrees=-30)
        elif action_index == 5:  # rotateright_small - Rotate right by 6 degrees
            self._last_event = self.env.step(action="RotateAgent", degrees=6)
        elif action_index == 6:  # rotateleft_small - Rotate left by 6 degrees
            self._last_event = self.env.step(action="RotateAgent", degrees=-6)
        elif action_index == 7:  # lookup - Deprecated for Stretch
            self._last_event = self.env.step(action="Pass")  # No-op for Stretch
        elif action_index == 8:  # lookdown - Deprecated for Stretch
            self._last_event = self.env.step(action="Pass")  # No-op for Stretch
        elif action_index == 9:  # pickup - Grasp action
            self._last_event = self.env.step(action="PickupObject")
        elif action_index == 10:  # dropoff - Release action
            self._last_event = self.env.step(action="ReleaseObject")
        elif action_index == 11:  # move_arm_up - Move arm up by 0.1m
            self._last_event = self.env.step(action="MoveArmRelative", y=0.1)
        elif action_index == 12:  # move_arm_down - Move arm down by 0.1m
            self._last_event = self.env.step(action="MoveArmRelative", y=-0.1)
        elif action_index == 13:  # move_arm_out - Extend arm out by 0.1m
            self._last_event = self.env.step(action="MoveArmRelative", z=0.1)
        elif action_index == 14:  # move_arm_in - Retract arm in by 0.1m
            self._last_event = self.env.step(action="MoveArmRelative", z=-0.1)
        elif action_index == 15:  # wrist_open - Rotate wrist counterclockwise 10°
            self._last_event = self.env.step(action="RotateWristRelative", yaw=-10)
        elif action_index == 16:  # wrist_close - Rotate wrist clockwise 10°
            self._last_event = self.env.step(action="RotateWristRelative", yaw=10)
        elif action_index == 17:  # move_arm_up_small - Move arm up by 0.02m
            self._last_event = self.env.step(action="MoveArmRelative", y=0.02)
        elif action_index == 18:  # move_arm_down_small - Move arm down by 0.02m
            self._last_event = self.env.step(action="MoveArmRelative", y=-0.02)
        elif action_index == 19:  # move_arm_out_small - Extend arm out by 0.02m
            self._last_event = self.env.step(action="MoveArmRelative", z=0.02)
        elif action_index == 20:  # move_arm_in_small - Retract arm in by 0.02m
            self._last_event = self.env.step(action="MoveArmRelative", z=-0.02)
        else:
            # Invalid action, do nothing
            self._last_event = self.env.step(action="Pass")
    
    def measure_success(self):
        """Check if the agent has successfully completed the Fetch task.
        
        For Fetch tasks, success means:
        1. The target object is being held by the agent
        2. OR the agent is very close to the target position
        
        Returns:
            success: Boolean indicating success
            distance: Distance to the target
        """
        agent_position = self.env.last_event.metadata["agent"]["position"]
        target_position = self.episode_data["target_position"]
        
        # Calculate distance to target
        dist = math.sqrt(
            (agent_position["x"] - target_position["x"])**2 +
            (agent_position["z"] - target_position["z"])**2
        )
        
        # Check if agent is holding the target object
        held_objects = []
        try:
            # For Stretch robot, check if any object is being held
            arm_metadata = self.env.last_event.metadata.get("arm", {})
            held_objects = arm_metadata.get("heldObjects", [])
        except:
            # Fallback: check general agent metadata
            try:
                agent_metadata = self.env.last_event.metadata.get("agent", {})
                held_objects = agent_metadata.get("heldObjects", [])
            except:
                held_objects = []
        
        # Check if target object is being held
        target_object_id = self.episode_data.get("targetObjectId", "")
        target_object_type = self.episode_data.get("targetObjectType", "")
        
        object_held = False
        if held_objects:
            # Check if any held object matches target
            for held_obj in held_objects:
                if (target_object_id and held_obj == target_object_id) or \
                   (target_object_type and target_object_type.lower() in held_obj.lower()):
                    object_held = True
                    break
        
        # Success conditions:
        # 1. Object is held by the agent, OR
        # 2. Agent is very close to target (within success_threshold)
        success = object_held or (dist <= self.success_threshold)
        
        return float(success), dist
    
    def _get_arm_state(self):
        """Get the current arm state for proprioception.
        
        Returns:
            str: Formatted arm state string
        """
        # Get arm position from agent metadata
        agent_metadata = self.env.last_event.metadata.get("agent", {}) if self.env.last_event.metadata else {}
        
        # Extract arm information (using mock data for now since AI2-THOR Stretch integration is in progress)
        # In a real implementation, this would come from agent_metadata["arm"] or similar
        z_pos = getattr(self, '_arm_z', 0.0)  # Extension (default retracted)
        y_pos = getattr(self, '_arm_y', 0.8)  # Height (default position)
        wrist_angle = getattr(self, '_wrist_angle', 0)  # Wrist rotation
        
        # Determine gripper state
        if hasattr(self, 'is_holding') and self.is_holding:
            # Try to get the held object name from metadata
            objects = self.env.last_event.metadata.get("objects", []) if self.env.last_event.metadata else []
            held_objects = [obj for obj in objects if obj.get("isPickedUp", False)]
            if held_objects:
                object_name = held_objects[0]["objectType"].lower()
                gripper_state = f"holding_{object_name}"
            else:
                gripper_state = "holding_object"
        else:
            gripper_state = "empty"
        
        return f"z={z_pos:.1f}m, y={y_pos:.1f}m, wrist={wrist_angle}°, gripper={gripper_state}"

    def _render(self, init_obs=True):
        """Render the environment observation.
        
        This method creates either a text representation or an image of the environment
        state, depending on the configured render mode. It formats the observation string
        based on whether this is the initial observation or a subsequent one.
        
        Args:
            init_obs: Whether this is the initial observation
            
        Returns:
            Observation dict
        """
        img_placeholder = self.config.get("image_placeholder", "<image>")
        
        # Get format prompt without examples for action/init templates
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False  # No examples for action and init obs
        )
        
        # ----- 文本模式：不传输真实图片，只保留文字描述 -----
        multi_modal_data = {img_placeholder: []}
        
        # Get current arm state
        arm_state = self._get_arm_state()
        
        # Format the template
        if init_obs:
            obs_str = init_observation_template(
                observation=img_placeholder,
                instruction=self.episode_language_instruction,
                arm_state=arm_state
            ) + "\n" + format_prompt_text
        else:
            obs_str = action_template(
                valid_action=self.valid_actions,
                observation=img_placeholder,
                reward=self.reward,
                done=self.measure_success()[0],
                instruction=self.episode_language_instruction,
                env_feedback=getattr(self, 'info', {}).get("env_feedback", "No feedback"),
                arm_state=arm_state
            ) + "\n" + format_prompt_text
        
        # ----------------  替换 <image> 为文本占位符 ----------------
        if isinstance(img_placeholder, str) and img_placeholder in obs_str:
            obs_str = obs_str.replace(img_placeholder, "[VISUAL]")
 
        return {
            "obs_str": obs_str,
            "multi_modal_data": multi_modal_data,
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
        
    
        return system_prompt(format=self.config.prompt_format) + '\n' + format_prompt_text
    
    def close(self):
        """Close the environment."""
        self.env.stop()
        
    def get_env_state(self):
        """
        Get the current state of the navigation environment focusing on visible objects.
        
        Returns:
            Dict: Contains target position, target direction, visible objects,
                and instruction information with rounded distances
        """
        # Get agent information
        agent_metadata = self.env.last_event.metadata["agent"]
        agent_position = agent_metadata["position"]
        agent_rotation = agent_metadata["rotation"]["y"]  # Only y-axis rotation is relevant
        
        # Get target information
        target_position = self.episode_data["target_position"]
        target_type = self.episode_data["targetObjectType"]
        success, distance = self.measure_success()
        
        # Calculate target's relative direction
        dx_target = target_position["x"] - agent_position["x"]
        dz_target = target_position["z"] - agent_position["z"]
        angle_to_target = math.degrees(math.atan2(dx_target, dz_target))
        relative_angle_target = (angle_to_target - agent_rotation) % 360
        if relative_angle_target > 180:
            relative_angle_target -= 360
            
        # Determine target's relative position
        if -45 <= relative_angle_target <= 45:
            target_relative_direction = "ahead"
        elif 45 < relative_angle_target <= 135:
            target_relative_direction = "right"
        elif -135 <= relative_angle_target < -45:
            target_relative_direction = "left"
        else:
            target_relative_direction = "back"
        
        # Get visible objects with position and relationship data
        objects = self.env.last_event.metadata["objects"]
        visible_objects = []
        
        for obj in objects:
            if obj.get("visible", False):
                obj_position = obj["position"]
                
                # Calculate distance from agent to object
                obj_distance = math.sqrt(
                    (agent_position["x"] - obj_position["x"])**2 +
                    (agent_position["z"] - obj_position["z"])**2
                )
                
                # Round distance to 2 decimal places
                obj_distance = round(obj_distance, 2)
                
                # Calculate relative angle to object (in degrees)
                dx = obj_position["x"] - agent_position["x"]
                dz = obj_position["z"] - agent_position["z"]
                angle_to_obj = math.degrees(math.atan2(dx, dz))
                # Adjust for agent's rotation (0 means directly in front)
                relative_angle = (angle_to_obj - agent_rotation) % 360
                if relative_angle > 180:
                    relative_angle -= 360
                    
                # Determine relative position (front, back, left, right)
                if -45 <= relative_angle <= 45:
                    relative_direction = "ahead"
                elif 45 < relative_angle <= 135:
                    relative_direction = "right"
                elif -135 <= relative_angle < -45:
                    relative_direction = "left"
                else:
                    relative_direction = "back"
                
                # Store object information
                visible_objects.append({
                    "type": obj["objectType"],
                    "direction_to_player": relative_direction,
                    "distance_to_player": obj_distance,
                })
    
        # Sort objects by distance (closest first)
        visible_objects.sort(key=lambda x: x["distance_to_player"])
        
        return {
            "target_obj_type": target_type, 
            "target_distance_to_player": round(distance, 2), 
            "target_direction_to_player": target_relative_direction,
            "visible_objects": visible_objects[:self.config.max_objects_in_state],   
        }


if __name__ == "__main__":
    # Example usage
    import os
    config = SpocEnvConfig()
    env = SpocEnv(config)
    print(env.system_prompt())
    
    obs, info = env.reset(seed=3)
    print(obs["obs_str"])
    i = 0
    os.makedirs("./test_navigation", exist_ok=True)
    img = obs["multi_modal_data"][config.image_placeholder][0]
    img.save(f"./test_navigation/navigation_{i}.png")
    done = False
    
    # Interactive testing loop
    while not done:
        i += 1
        action = input("Enter action (moveahead, moveback, moveright, moveleft, rotateright, rotateleft, lookup, lookdown): ")
        action = f"<think>Let me navigate toward the target.</think><answer>{action}</answer>"
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}")
        print(obs["obs_str"])
        img = obs["multi_modal_data"][config.image_placeholder][0]
        img.save(f"./test_navigation/navigation_{i}.png")
        print(f"Success: {info['metrics']['traj_metrics']['success']}, Action effective: {info['metrics']['turn_metrics']['action_is_effective']}")
        
        if done:
            break
    
    print(f"Total reward: {env.compute_reward()}")
    env.close()