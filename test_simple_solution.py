#!/usr/bin/env python3
"""
Simple test to see if we can solve the issue without sudo.
Sometimes the issue is just configuration, not missing X server.
"""

import os
import subprocess

def test_simple_fixes():
    """Test simple fixes that don't require sudo."""
    print("Testing Simple Solutions (No Sudo Required)")
    print("="*60)
    
    # Test 1: Set DISPLAY to a working value
    print("\nTest 1: Setting DISPLAY environment variable")
    original_display = os.environ.get('DISPLAY')
    
    try:
        # Unset DISPLAY first
        if 'DISPLAY' in os.environ:
            del os.environ['DISPLAY']
        
        import ai2thor.controller
        controller = ai2thor.controller.Controller(
            platform="CloudRendering",
            headless=True,
            width=224,
            height=224,
            agentMode="stretch"
        )
        
        event = controller.reset(scene="FloorPlan1_physics")
        
        if event.frame is not None:
            print(f"✓ CloudRendering works without DISPLAY! Frame: {event.frame.shape}")
            controller.stop()
            return True
        else:
            print("✗ CloudRendering still returns None frame")
            
        controller.stop()
        
    except Exception as e:
        print(f"✗ CloudRendering failed: {e}")
    
    # Test 2: Try with minimal DISPLAY
    print("\nTest 2: Testing with different DISPLAY values")
    for display_val in [':0', ':1', ':99', None]:
        try:
            if display_val is None:
                if 'DISPLAY' in os.environ:
                    del os.environ['DISPLAY']
                print("  Testing with no DISPLAY variable")
            else:
                os.environ['DISPLAY'] = display_val
                print(f"  Testing with DISPLAY={display_val}")
            
            controller = ai2thor.controller.Controller(
                platform="CloudRendering",
                headless=True,
                width=224,
                height=224
            )
            
            event = controller.reset(scene="FloorPlan1_physics")
            controller.stop()
            
            if event.frame is not None:
                print(f"    ✓ Works! Frame: {event.frame.shape}")
                return True
            else:
                print(f"    ✗ Frame still None")
                
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    
    # Restore original DISPLAY
    if original_display:
        os.environ['DISPLAY'] = original_display
    elif 'DISPLAY' in os.environ:
        del os.environ['DISPLAY']
    
    return False

def test_software_rendering():
    """Test with software rendering environment variables."""
    print(f"\nTest 3: Software rendering configuration")
    
    # Set software rendering variables
    env_vars = {
        'MESA_GL_VERSION_OVERRIDE': '3.3',
        'MESA_GLSL_VERSION_OVERRIDE': '330',
        'LIBGL_ALWAYS_SOFTWARE': '1',
        'GALLIUM_DRIVER': 'softpipe',
        'PYOPENGL_PLATFORM': 'egl'
    }
    
    # Save original values
    original_env = {}
    for key in env_vars:
        original_env[key] = os.environ.get(key)
        os.environ[key] = env_vars[key]
    
    try:
        import ai2thor.controller
        controller = ai2thor.controller.Controller(
            platform="CloudRendering",
            headless=True,
            width=224,
            height=224
        )
        
        event = controller.reset(scene="FloorPlan1_physics")
        controller.stop()
        
        if event.frame is not None:
            print(f"✓ Software rendering works! Frame: {event.frame.shape}")
            return True
        else:
            print("✗ Software rendering - frame still None")
            
    except Exception as e:
        print(f"✗ Software rendering failed: {e}")
    
    finally:
        # Restore environment
        for key, value in original_env.items():
            if value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = value
    
    return False

def test_spoc_if_working():
    """Test SPOC if we found a working solution."""
    print(f"\n{'='*60}")
    print("TESTING SPOC WITH WORKING SOLUTION")
    print(f"{'='*60}")
    
    try:
        import sys
        sys.path.insert(0, '/home/jiajunliu/VAGEN')
        os.environ['SPOC_DATA_PATH'] = '/home/jiajunliu/spoc_data/fifteen'
        
        from vagen.env.spoc import SpocEnv, SpocEnvConfig
        
        config = SpocEnvConfig(
            data_path=os.environ['SPOC_DATA_PATH'],
            task_type="FetchType", 
            chores_split="train",
            resolution=224,
            prompt_format="free_think"
        )
        
        env = SpocEnv(config)
        print("✓ SpocEnv initialized!")
        
        obs, info = env.reset(seed=42)
        print("✓ Reset successful!")
        print(f"Scene: {env.episode_data['scene']}")
        
        action_str = "<think>Testing</think><answer>moveahead</answer>"
        obs, reward, done, info = env.step(action_str)
        print("✓ Step successful!")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ SPOC test failed: {e}")
        return False

def main():
    print("Trying simple solutions first (no sudo required)...")
    
    success = (test_simple_fixes() or 
               test_software_rendering())
    
    if success:
        print(f"\n{'='*60}")
        print("FOUND A WORKING SOLUTION!")
        print(f"{'='*60}")
        
        spoc_works = test_spoc_if_working()
        
        if spoc_works:
            print("\n✓ COMPLETE SUCCESS! Your SPOC environment works!")
            print("\nNo need for sudo or complex X server setup.")
            print("The issue was just configuration, not missing components.")
        else:
            print("\nAI2-THOR works but SPOC integration needs more work.")
    else:
        print(f"\n{'='*60}")
        print("SIMPLE SOLUTIONS FAILED")
        print(f"{'='*60}")
        print("You'll need to use the ai2thor-xorg solution:")
        print("1. Run: python /home/jiajunliu/VAGEN/test_ai2thor_xorg.py")
        print("2. Enter your sudo password when prompted")
        print("3. This will set up the proper X11 server")

if __name__ == "__main__":
    main()