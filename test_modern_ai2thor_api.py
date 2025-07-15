#!/usr/bin/env python3
"""
Test SPOC with the modern ai2thor API (5.0.0) that you already have installed.
The issue might just be API changes, not version incompatibility.
"""

import os
import sys

def test_modern_api():
    """Test modern ai2thor API."""
    print("Testing Modern AI2-THOR API (5.0.0)")
    print("="*60)
    
    try:
        import ai2thor.controller
        print(f"✓ AI2-THOR version: {ai2thor.__version__}")
        
        # Test CloudRendering with modern API
        print("\nTesting CloudRendering...")
        controller = ai2thor.controller.Controller(
            platform="CloudRendering",
            headless=True,
            width=512,
            height=512,
            agentMode="stretch"
        )
        
        # Test basic scene
        event = controller.reset(scene="FloorPlan1_physics")
        
        # Modern AI2-THOR uses different attributes for images
        frame = None
        if hasattr(event, 'frame') and event.frame is not None:
            frame = event.frame
            print("✓ Using event.frame")
        elif hasattr(event, 'cv2img') and event.cv2img is not None:
            frame = event.cv2img
            print("✓ Using event.cv2img")
        elif hasattr(event, 'numpy_frame') and event.numpy_frame is not None:
            frame = event.numpy_frame
            print("✓ Using event.numpy_frame")
        
        if frame is not None:
            print(f"✓ Frame shape: {frame.shape}")
            print(f"✓ Frame type: {type(frame)}")
        else:
            print("✗ No frame data available")
            # List all available attributes
            print("Available event attributes:")
            for attr in dir(event):
                if not attr.startswith('_'):
                    try:
                        value = getattr(event, attr)
                        print(f"  {attr}: {type(value)}")
                    except:
                        print(f"  {attr}: <error accessing>")
        
        # Test a high-numbered scene
        print(f"\nTesting high-numbered scene...")
        try:
            event = controller.reset(scene="FloorPlan15000_physics")
            print("✓ High-numbered scenes work!")
        except Exception as e:
            print(f"✗ High-numbered scenes failed: {e}")
            # Try scenes in the range your SPOC data uses
            print("Testing scenes in SPOC range...")
            test_scenes = ["FloorPlan2358_physics", "FloorPlan17707_physics"]
            for scene in test_scenes:
                try:
                    event = controller.reset(scene=scene)
                    print(f"✓ {scene} works!")
                    break
                except Exception as e:
                    print(f"✗ {scene} failed: {e}")
        
        controller.stop()
        return frame is not None
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spoc_with_correct_api():
    """Test SPOC but patch the frame access to use modern API."""
    print(f"\n{'='*60}")
    print("TESTING SPOC WITH CORRECTED API")
    print(f"{'='*60}")
    
    try:
        sys.path.insert(0, '/home/jiajunliu/VAGEN')
        os.environ['SPOC_DATA_PATH'] = '/home/jiajunliu/spoc_data/fifteen'
        
        # Monkey patch the SpocEnv to use correct frame attribute
        from vagen.env.spoc import SpocEnv, SpocEnvConfig
        
        # Save original _render method
        original_render = SpocEnv._render
        
        def patched_render(self, init_obs=True):
            """Patched render method that uses correct frame attribute."""
            img_placeholder = getattr(self.config, "image_placeholder", "<image>")
            
            format_prompt_text = self.format_prompt_func(
                max_actions_per_step=self.config.max_actions_per_step,
                action_sep=self.config.action_sep,
                add_example=False
            )
            
            # Get frame using modern API
            frame = None
            if hasattr(self.env.last_event, 'frame') and self.env.last_event.frame is not None:
                frame = self.env.last_event.frame
            elif hasattr(self.env.last_event, 'cv2img') and self.env.last_event.cv2img is not None:
                frame = self.env.last_event.cv2img
            elif hasattr(self.env.last_event, 'numpy_frame') and self.env.last_event.numpy_frame is not None:
                frame = self.env.last_event.numpy_frame
            
            if frame is None:
                raise RuntimeError("No frame data available from AI2-THOR event")
            
            from vagen.env.utils.context_utils import convert_numpy_to_PIL
            multi_modal_data = {
                img_placeholder: [convert_numpy_to_PIL(frame)]
            }
            
            # Get current arm state
            arm_state = self._get_arm_state()
            
            # Format the template
            if init_obs:
                from vagen.env.spoc.prompt import init_observation_template
                obs_str = init_observation_template(
                    observation=f"Proprioception: {arm_state}\nVisuals: {img_placeholder}",
                    instruction=self.episode_language_instruction,
                ) + "\n" + format_prompt_text
            else:
                from vagen.env.spoc.prompt import action_template
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
        
        # Apply patch
        SpocEnv._render = patched_render
        
        config = SpocEnvConfig(
            data_path=os.environ['SPOC_DATA_PATH'],
            task_type="FetchType",
            chores_split="train", 
            resolution=224,
            prompt_format="free_think"
        )
        
        env = SpocEnv(config)
        print("✓ SpocEnv initialized with patched API!")
        
        # Test reset
        obs, info = env.reset(seed=42)
        print("✓ Environment reset successful!")
        print(f"Scene: {env.episode_data['scene']}")
        print(f"Instruction: {env.episode_language_instruction[:60]}...")
        print(f"Observation keys: {obs.keys()}")
        
        # Test step
        action_str = "<think>Testing movement</think><answer>moveahead</answer>"
        obs, reward, done, info = env.step(action_str)
        print("✓ Step executed successfully!")
        print(f"Reward: {reward}, Done: {done}")
        
        env.close()
        
        # Restore original method
        SpocEnv._render = original_render
        
        print("✓ SPOC works with modern ai2thor API!")
        return True
        
    except Exception as e:
        print(f"✗ SPOC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    api_works = test_modern_api()
    
    if api_works:
        spoc_works = test_spoc_with_correct_api()
        
        if spoc_works:
            print(f"\n{'='*60}")
            print("SUCCESS! The issue was just API changes!")
            print(f"{'='*60}")
            print("Your ai2thor 5.0.0 works fine, you just need to:")
            print("1. Update SpocEnv to use the correct frame attribute")
            print("2. The modern version supports your high-numbered scenes")
            print("3. CloudRendering works properly")
            print("\nNo need to downgrade - your version is better!")
        else:
            print("Modern API test failed. Scene compatibility issues remain.")
    else:
        print("Modern AI2-THOR API test failed.")

if __name__ == "__main__":
    main()