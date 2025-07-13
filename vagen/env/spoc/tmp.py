# =========================================================================================
# 请将 SpocEnv 类的所有代码替换为以下内容
# =========================================================================================
class SpocEnv(BaseEnv):
    """SPOC environment with Stretch robot manipulation tasks."""

    ValidEvalSets = [
        'base', 'common_sense', 'complex_instruction', 'visual_appearance', 'long_horizon'
    ]

    # <<< 动作字典保持不变 >>>
    ACTION_LOOKUP = {
        # ... (所有动作)
        "moveahead": 1, "moveback": 2, "rotateright": 3, "rotateleft": 4, 
        "rotateright_small": 5, "rotateleft_small": 6, "lookup": 7, "lookdown": 8,
        "pickup": 9, "dropoff": 10, "move_arm_up": 11, "move_arm_down": 12,
        "move_arm_out": 13, "move_arm_in": 14, "wrist_open": 15, "wrist_close": 16,
        "move_arm_up_small": 17, "move_arm_down_small": 18, "move_arm_out_small": 19, "move_arm_in_small": 20,
    }

    DISCRETE_SKILLSET = [
        # ... (所有动作描述)
    ]

    def __init__(self, config: SpocEnvConfig):
        # <<< __init__ 方法保持不变, 您的版本是正确的 >>>
        super().__init__()
        self.config = config
        
        env_vars_to_set = {
            'XAUTHORITY': '',
            'XDG_RUNTIME_DIR': '/tmp',
            'GALLIUM_DRIVER': 'softpipe',
            'MESA_GL_VERSION_OVERRIDE': '3.3', 'LIBGL_ALWAYS_SOFTWARE': '1',
            'LIBGL_ALWAYS_INDIRECT': '1', 'EGL_PLATFORM': 'surfaceless', 'PYOPENGL_PLATFORM': 'egl',
        }
        for key, value in env_vars_to_set.items():
            os.environ[key] = value

        self.thor_config = {
            "agentMode": "stretch",
            "gridSize": 0.1,
            "visibilityDistance": 10,
            "renderDepthImage": False,
            "renderInstanceSegmentation": False,
            "width": config.resolution,
            "height": config.resolution,
            "fieldOfView": config.fov,
            "server_timeout": 900, "server_start_timeout": 900, "quality": "Low",    
        }

        self.env = None
        platforms_to_try = ["CloudRendering", "Linux64"]

        for platform_str in platforms_to_try:
            try:
                config_to_try = {**self.thor_config, "platform": platform_str, "headless": True}
                print(f"Attempting AI2-THOR with platform: {platform_str}")
                self.env = ai2thor.controller.Controller(**config_to_try)
                print(f"Successfully initialized AI2-THOR with platform: {platform_str}")
                self.thor_config = config_to_try
                break
            except Exception as e:
                print(f"Failed to initialize with platform {platform_str}: {e}")
                if self.env:
                    try: self.env.stop()
                    except Exception: pass
                    self.env = None
        
        if self.env is None:
            raise RuntimeError("Failed to initialize AI2-THOR with any platform configuration.")
        
        self.dataset = get_dataset(
            data_path=config.data_path, 
            task_type=config.task_type, 
            split=config.chores_split
        )
        print(f"[SpocEnv] Loaded {len(self.dataset)} episodes")
        self.number_of_episodes = len(self.dataset)
        
        self._current_episode_num = 0
        self._current_step = 0
        self._max_episode_steps = 100
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
        # <<< CORRECTED LOGIC: 这是一个健壮的 reset 版本 >>>
        """Reset the environment to a new episode from the real dataset."""
        if self.number_of_episodes == 0:
            raise RuntimeError("Cannot reset environment, SPOC dataset is empty or not found.")

        if seed is not None:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, self.number_of_episodes)
        else:
            idx = np.random.randint(0, self.number_of_episodes)
        
        self._current_episode_num = idx
        
        traj_data = self.dataset[idx]
        self.episode_data = traj_data
        self.episode_language_instruction = traj_data["instruction"]

        # 在 try 块外部定义 scene_name，确保它在 try 块中可用
        scene_name = traj_data["scene"]
        
        try:
            # 步骤1：重置场景，并立刻检查是否成功
            self._last_event = self.env.reset(scene=scene_name)
            if not self._last_event or not self._last_event.metadata.get('lastActionSuccess'):
                raise RuntimeError(f"Failed to reset to scene {scene_name}. AI2-THOR controller returned failure.")

            # 步骤2：传送 agent，并立刻检查是否成功
            pose = traj_data["agentPose"]
            self._last_event = self.env.step(
                action="Teleport",
                position=pose["position"],
                rotation={'x': 0, 'y': pose["rotation"], 'z': 0},
                horizon=pose["horizon"],
                standing=True
            )
            if not self._last_event or not self._last_event.metadata.get('lastActionSuccess'):
                raise RuntimeError(f"Failed to teleport agent in scene {scene_name}. AI2-THOR controller returned failure.")

        except Exception as e:
            print(f"Critical error during SpocEnv.reset: {e}. This environment instance is now considered dead.")
            raise RuntimeError(f"Unrecoverable error in SpocEnv.reset: {e}") from e

        # --- 只有在上面全部成功后，才重置这些状态变量 ---
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
        # <<< CORRECTED LOGIC: 这是一个更智能的 step 版本 >>>
        """Execute an action in the environment."""
        print(f"[DEBUG SPOC] ===== Starting step() with action_str =====")
        print(f"[DEBUG SPOC] Raw LLM response: {action_str}")
        
        rst = self.parse_func(
            response=action_str,
            special_token_list=getattr(self.config, 'special_token_list', None),
            action_sep=getattr(self.config, 'action_sep', ',') or ',',
            max_actions=getattr(self.config, 'max_actions_per_step', 1) or 1
        )
        
        action_list = rst.get('actions', [])
        
        # 必须先检查 _last_event 是否有效，防止 reset 失败后的连锁崩溃
        if not self._last_event:
            raise RuntimeError("Cannot step with invalid or None self._last_event. The environment may have failed to reset.")
            
        prev_pos = self._last_event.metadata["agent"]["position"]
        
        done = False
        info = {}
        info.update(rst)
        info['llm_raw_response'] = action_str
        
        # ---- 动作执行逻辑 ----
        # 只要解析出了动作列表，就去执行
        if action_list:
            print(f"[DEBUG SPOC] Actions found. Executing {len(action_list)} actions: {action_list}")
            
            for action in action_list:
                action_lower = action.lower()
                if action_lower in self.ACTION_LOOKUP:
                    self._execute_action(self.ACTION_LOOKUP[action_lower])
                    self.valid_actions.append(action)
                    success, _ = self.measure_success()
                    if success:
                        self.reward += 10.0
                        done = True
                else:
                    print(f"[DEBUG SPOC] INVALID ACTION in list: '{action_lower}'")
                    break # 遇到无效动作，立即停止
                
                if done: break
            
            self._current_step += 1
            if self._current_step >= self._max_episode_steps:
                done = True
        else:
            # 如果没有解析出任何动作，执行 Pass 并等待下一轮
            print(f"[DEBUG SPOC] No valid actions found. Executing 'Pass' action.")
            self._execute_action(self.ACTION_LOOKUP['lookup'])

        # ---- 奖励逻辑 ----
        self.reward = 0.0 # 重置单步奖励
        if rst.get("format_correct", False):
            self.reward += getattr(self.config, 'format_reward', 0.0)
            info["is_format_rewarded"] = True
        else:
            info["is_format_rewarded"] = False

        # ---- 更新 info 字典 ----
        curr_pos = self._last_event.metadata["agent"]["position"]
        success, distance = self.measure_success()

        info["metrics"] = {
            "turn_metrics": {
                "action_is_valid": bool(action_list),
                "action_is_effective": curr_pos["x"] != prev_pos["x"] or curr_pos["z"] != prev_pos["z"]
            },
            "traj_metrics": { "success": success }
        }
        info['distance'] = distance
        info['instruction'] = self.episode_language_instruction
        info['env_step'] = self._current_step
        info['task_success'] = success
        info['last_action_success'] = self._last_event.metadata['lastActionSuccess']
        info["env_feedback"] ="Last action is executed successfully." if info['last_action_success'] else "Last action is not executed successfully."
        
        self.total_reward += self.reward
        
        return self._render(init_obs=False), self.reward, done, info

    def system_prompt(self):
        # <<< CORRECTED LOGIC: 启用更严格的 Prompt >>>
        """Get the system prompt for the environment."""
        spoc_system_prompt = (
            "You are an AI agent controlling a Stretch robot. Your ONLY task is to output actions to complete goals."
            "You MUST respond in the specified format. DO NOT generate any other text, conversation, or explanations."
            "First, think step-by-step inside the <think> tag. Analyze the observation and plan your next move."
            "Then, you MUST provide one or more valid actions from the list, separated by commas, in the <answer> tag."
            "\n\n**CRITICAL: Your response MUST strictly follow this format:**"
            "\n<think>Your detailed reasoning here.</think><answer>action_name1,action_name2</answer>"
            "\n\n**Example 1 (Single Action):**"
            "\n<think>The object is to my left, so I need to rotate left to face it.</think><answer>rotateleft</answer>"
            "\n\n**Example 2 (Multiple Actions):**"
            "\n<think>The object is far away. I will move forward twice.</think><answer>moveahead,moveahead</answer>"
            "\n\nNow, begin."
        )
        return spoc_system_prompt
    
    # --- 其他方法 (_execute_action, measure_success, _get_arm_state, _render, etc.) 保持不变 ---
    # ... 您可以从您现有的文件中复制这部分，它们是正确的 ...
    def _execute_action(self, action_index: int):
        action_map = {
            1: {"action": "MoveAhead", "moveMagnitude": 0.2}, 2: {"action": "MoveBack", "moveMagnitude": 0.2},
            3: {"action": "RotateRight", "degrees": 30}, 4: {"action": "RotateLeft", "degrees": 30},
            5: {"action": "RotateRight", "degrees": 6}, 6: {"action": "RotateLeft", "degrees": 6},
            7: {"action": "Pass"}, 8: {"action": "Pass"}, 9: {"action": "PickupObject"},
            10: {"action": "ReleaseObject"}, 11: {"action": "MoveArmRelative", "y": 0.1},
            12: {"action": "MoveArmRelative", "y": -0.1}, 13: {"action": "MoveArmRelative", "z": 0.1},
            14: {"action": "MoveArmRelative", "z": -0.1}, 15: {"action": "RotateWristRelative", "yaw": -10},
            16: {"action": "RotateWristRelative", "yaw": 10}, 17: {"action": "MoveArmRelative", "y": 0.02},
            18: {"action": "MoveArmRelative", "y": -0.02}, 19: {"action": "MoveArmRelative", "z": 0.02},
            20: {"action": "MoveArmRelative", "z": -0.02},
        }
        params = action_map.get(action_index, {"action": "Pass"})
        self._last_event = self.env.step(**params)
        if params["action"] == "PickupObject" and self._last_event.metadata["lastActionSuccess"]: self.is_holding = True
        elif params["action"] == "ReleaseObject" and self._last_event.metadata["lastActionSuccess"]: self.is_holding = False

    def measure_success(self):
        success = False
        try:
            if self.is_holding and self.episode_data:
                held_objects = self.env.last_event.metadata['arm']['heldObjects']
                target_type = self.episode_data.get("targetObjectType")
                if held_objects and target_type:
                    for obj in held_objects:
                        if obj['objectType'].startswith(target_type):
                            success = True; break
        except (KeyError, IndexError) as e:
            print(f"Warning: Could not check success due to missing metadata: {e}"); success = False
        agent_pos = self.env.last_event.metadata["agent"]["position"]
        target_pos = self.episode_data.get("target_position", agent_pos) if self.episode_data else agent_pos
        distance = math.sqrt((agent_pos["x"] - target_pos["x"])**2 + (agent_pos["z"] - target_pos["z"])**2)
        return float(success), distance
    
    def _get_arm_state(self) -> str:
        try:
            arm_meta = self.env.last_event.metadata['arm']; arm_pos = arm_meta['position']; held_objects = arm_meta['heldObjects']
            if held_objects: gripper_state = f"holding a {held_objects[0]['objectType'].split('_')[0].lower()}"
            else: gripper_state = "empty"
            return f"Arm is at (y={arm_pos['y']:.2f}, z={arm_pos['z']:.2f}), gripper is {gripper_state}."
        except (KeyError, IndexError): return "Arm state is unavailable."

    def _render(self, init_obs=True):
        img_placeholder = getattr(self.config, "image_placeholder", "<image>")
        frame = self.env.last_event.frame
        multi_modal_data = { img_placeholder: [convert_numpy_to_PIL(frame)] }
        arm_state = self._get_arm_state()
        if init_obs:
            obs_str = init_observation_template(
                observation=f"Proprioception: {arm_state}\nVisuals: {img_placeholder}",
                instruction=self.episode_language_instruction,
            )
        else:
            obs_str = action_template(
                valid_action=self.valid_actions,
                observation=f"Proprioception: {arm_state}\nVisuals: {img_placeholder}",
                reward=self.reward, done=self.measure_success()[0],
                instruction=self.episode_language_instruction,
                env_feedback=self.info["env_feedback"]
            )
        return { "obs_str": obs_str, "multi_modal_data": multi_modal_data }
    
    def close(self):
        self.env.stop()
        
    def get_env_state(self):
        # This method seems unused in the main loop but is good for debugging
        return {} # Placeholder
# =========================================================================================
# 替换结束
# =========================================================================================