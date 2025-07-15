#!/usr/bin/env python3
"""
Test different AI2-THOR rendering configurations to find one that works.
"""

import os

def test_rendering_configs():
    """Test various AI2-THOR rendering configurations."""
    print("Testing AI2-THOR 5.0.0 Rendering Configurations")
    print("="*60)
    
    import ai2thor.controller
    
    configs_to_test = [
        {
            "name": "CloudRendering (headless)",
            "config": {
                "platform": "CloudRendering",
                "headless": True,
                "width": 224,
                "height": 224,
                "agentMode": "stretch"
            }
        },
        {
            "name": "CloudRendering (with renderDepthImage)",
            "config": {
                "platform": "CloudRendering", 
                "headless": True,
                "width": 224,
                "height": 224,
                "agentMode": "stretch",
                "renderDepthImage": True,
                "renderInstanceSegmentation": True
            }
        },
        {
            "name": "Linux64 (headless)",
            "config": {
                "platform": "Linux64",
                "headless": True,
                "width": 224,
                "height": 224,
                "agentMode": "stretch"
            }
        },
        {
            "name": "OSXIntel64 (if available)",
            "config": {
                "platform": "OSXIntel64",
                "headless": True,
                "width": 224,
                "height": 224,
                "agentMode": "stretch"
            }
        },
        {
            "name": "Auto platform detection",
            "config": {
                "headless": True,
                "width": 224,
                "height": 224,
                "agentMode": "stretch"
            }
        },
        {
            "name": "Default configuration",
            "config": {
                "agentMode": "stretch",
                "width": 224,
                "height": 224
            }
        }
    ]
    
    working_configs = []
    
    for test_config in configs_to_test:
        print(f"\nTesting: {test_config['name']}")
        controller = None
        try:
            controller = ai2thor.controller.Controller(**test_config['config'])
            event = controller.reset(scene="FloorPlan1_physics")
            
            if event.frame is not None:
                print(f"  ✓ SUCCESS - Frame shape: {event.frame.shape}")
                working_configs.append(test_config)
            else:
                print(f"  ✗ Frame is None")
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        finally:
            if controller:
                try:
                    controller.stop()
                except:
                    pass
    
    return working_configs

def test_spoc_with_working_config(working_configs):
    """Test SPOC with the first working configuration."""
    if not working_configs:
        print("\nNo working rendering configurations found!")
        return False
    
    print(f"\n{'='*60}")
    print(f"TESTING SPOC WITH WORKING CONFIG")
    print(f"{'='*60}")
    
    import sys
    sys.path.insert(0, '/home/jiajunliu/VAGEN')
    os.environ['SPOC_DATA_PATH'] = '/home/jiajunliu/spoc_data/fifteen'
    
    # Use the first working config
    best_config = working_configs[0]
    print(f"Using config: {best_config['name']}")
    
    try:
        # Patch the SpocEnv to use our working configuration
        from vagen.env.spoc.env import SpocEnv
        from vagen.env.spoc import SpocEnvConfig
        
        # Save original __init__
        original_init = SpocEnv.__init__
        
        def patched_init(self, config):
            """Patched init that uses working AI2-THOR config."""
            # Call parent init
            from vagen.env.base.base_env import BaseEnv
            BaseEnv.__init__(self)
            self.config = config
            
            # Set environment variables
            env_vars_to_set = {
                'XAUTHORITY': '',
                'XDG_RUNTIME_DIR': '/tmp',
            }
            for key, value in env_vars_to_set.items():
                os.environ[key] = value
            
            # Use our working thor config
            import ai2thor.controller
            self.thor_config = {
                **best_config['config'],
                "visibilityDistance": 10,
                "renderDepthImage": False,
                "renderInstanceSegmentation": False,
                "width": config.resolution,
                "height": config.resolution,
                "fieldOfView": config.fov,
                "server_timeout": 900,
                "server_start_timeout": 900,
                "quality": "Low",
            }
            
            self.env = ai2thor.controller.Controller(**self.thor_config)
            print(f"✓ AI2-THOR initialized with {best_config['name']}")
            
            # Load dataset
            from vagen.env.spoc.task_loader import get_dataset
            self.dataset = get_dataset(
                data_path=config.data_path,
                task_type=config.task_type,
                split=config.chores_split
            )
            print(f"[SpocEnv] Loaded {len(self.dataset)} episodes")
            self.number_of_episodes = len(self.dataset)
            
            # Initialize other attributes (from original code)
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
            
            from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
            from vagen.env.spoc.prompt import format_prompt
            self.format_prompt_func = format_prompt[self.config.prompt_format]
            self.parse_func = PARSE_FUNC_MAP[self.config.prompt_format]
        
        # Apply patch
        SpocEnv.__init__ = patched_init
        
        config = SpocEnvConfig(
            data_path=os.environ['SPOC_DATA_PATH'],
            task_type="FetchType",
            chores_split="train",
            resolution=224,
            prompt_format="free_think"
        )
        
        env = SpocEnv(config)
        print("✓ SpocEnv initialized successfully!")
        
        # Test reset
        obs, info = env.reset(seed=42)
        print("✓ Environment reset successful!")
        print(f"Scene: {env.episode_data['scene']}")
        print(f"Instruction: {env.episode_language_instruction[:60]}...")
        
        # Test step
        action_str = "<think>Testing movement</think><answer>moveahead</answer>"
        obs, reward, done, info = env.step(action_str)
        print("✓ Step executed successfully!")
        
        env.close()
        
        # Restore original
        SpocEnv.__init__ = original_init
        
        print(f"\n✓ SUCCESS! SPOC works with {best_config['name']}!")
        return True
        
    except Exception as e:
        print(f"✗ SPOC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    working_configs = test_rendering_configs()
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if working_configs:
        print("Working configurations found:")
        for config in working_configs:
            print(f"  ✓ {config['name']}")
        
        # Test SPOC with the best one
        spoc_success = test_spoc_with_working_config(working_configs)
        
        if spoc_success:
            print(f"\n{'='*60}")
            print("SOLUTION FOUND!")
            print(f"{'='*60}")
            print(f"Use the following configuration in your SpocEnv:")
            print(f"Config: {working_configs[0]['config']}")
            print("\nYour ai2thor 5.0.0 works fine - just needed the right configuration!")
        
    else:
        print("✗ No working rendering configurations found.")
        print("This suggests a deeper issue with AI2-THOR installation or system setup.")

if __name__ == "__main__":
    main()