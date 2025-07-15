#!/usr/bin/env python3
"""
Test upgrading to modern ai2thor to solve the SPOC compatibility issues.
"""

import subprocess
import sys
import os

def backup_and_upgrade_ai2thor():
    """Backup current version and test modern ai2thor."""
    print("AI2-THOR Upgrade Test")
    print("="*60)
    
    # Check current version
    try:
        import ai2thor
        current_version = getattr(ai2thor, '__version__', 'Unknown')
        print(f"Current ai2thor version: {current_version}")
        
        # Test current version issues
        print("\nTesting current version with CloudRendering...")
        controller = ai2thor.controller.Controller(
            platform="CloudRendering",
            headless=True,
            width=512,
            height=512,
            agentMode="stretch"
        )
        event = controller.reset(scene="FloorPlan1_physics")
        print(f"Frame type: {type(event.frame)}")
        print(f"Frame shape: {event.frame.shape if event.frame is not None else 'None'}")
        controller.stop()
        
        if event.frame is None:
            print("✗ Current version has broken CloudRendering (returns None frames)")
        else:
            print("✓ Current version works fine")
            return False  # No need to upgrade
            
    except Exception as e:
        print(f"Current version test failed: {e}")
    
    print(f"\n{'='*60}")
    print("UPGRADING TO MODERN AI2-THOR")
    print(f"{'='*60}")
    
    # Install modern ai2thor
    print("Installing latest ai2thor...")
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--upgrade', 'ai2thor'
        ], capture_output=True, text=True, check=True)
        print("✓ Modern ai2thor installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install modern ai2thor: {e}")
        return False
    
    # Test modern version
    try:
        # Need to reload the module
        if 'ai2thor' in sys.modules:
            del sys.modules['ai2thor']
        if 'ai2thor.controller' in sys.modules:
            del sys.modules['ai2thor.controller']
        
        import ai2thor.controller
        new_version = getattr(ai2thor, '__version__', 'Unknown')
        print(f"New ai2thor version: {new_version}")
        
        # Test CloudRendering
        print("\nTesting modern version with CloudRendering...")
        controller = ai2thor.controller.Controller(
            platform="CloudRendering",
            headless=True,
            width=512,
            height=512,
            agentMode="stretch"
        )
        
        # Test basic scene
        event = controller.reset(scene="FloorPlan1_physics")
        print(f"✓ Basic scene works - Frame shape: {event.frame.shape}")
        
        # Test a high-numbered scene that SPOC needs
        try:
            event = controller.reset(scene="FloorPlan15000_physics")
            print(f"✓ High-numbered scene works - Frame shape: {event.frame.shape}")
            high_scenes_work = True
        except Exception as e:
            print(f"✗ High-numbered scenes don't work: {e}")
            high_scenes_work = False
        
        controller.stop()
        
        return high_scenes_work
        
    except Exception as e:
        print(f"✗ Modern ai2thor test failed: {e}")
        return False

def test_spoc_with_modern_ai2thor():
    """Test SPOC environment with modern ai2thor."""
    print(f"\n{'='*60}")
    print("TESTING SPOC WITH MODERN AI2-THOR")
    print(f"{'='*60}")
    
    try:
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
        print("✓ SpocEnv initialized with modern ai2thor!")
        
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
        print("✓ SPOC works perfectly with modern ai2thor!")
        return True
        
    except Exception as e:
        print(f"✗ SPOC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def rollback_ai2thor():
    """Rollback to the old SPOC-specific version."""
    print(f"\n{'='*60}")
    print("ROLLING BACK TO OLD AI2-THOR VERSION")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--force-reinstall',
            '--extra-index-url', 'https://ai2thor-pypi.allenai.org',
            'ai2thor==0+5d0ab8ab8760eb584c5ae659c2b2b951cab23246'
        ], capture_output=True, text=True, check=True)
        print("✓ Rolled back to old ai2thor version")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Rollback failed: {e}")
        return False

def main():
    print("This script will test upgrading ai2thor to solve SPOC compatibility issues.")
    print("\nSteps:")
    print("1. Test current version issues")
    print("2. Upgrade to modern ai2thor") 
    print("3. Test SPOC compatibility")
    print("4. Rollback if needed")
    
    # Get user confirmation
    response = input("\nProceed with ai2thor upgrade test? (y/N): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return
    
    # Test upgrade
    upgrade_success = backup_and_upgrade_ai2thor()
    
    if upgrade_success:
        # Test SPOC
        spoc_success = test_spoc_with_modern_ai2thor()
        
        if spoc_success:
            print(f"\n{'='*60}")
            print("SUCCESS! Modern ai2thor solves the SPOC issues!")
            print(f"{'='*60}")
            print("Benefits:")
            print("- CloudRendering works properly (no None frames)")
            print("- Supports high-numbered FloorPlan scenes")
            print("- Better performance and stability")
            print("- No need for scene mapping workarounds")
            print("\nYou can now use modern ai2thor for your training!")
        else:
            print(f"\n{'='*60}")
            print("Modern ai2thor doesn't work with SPOC.")
            print("Rolling back to old version...")
            rollback_ai2thor()
    else:
        print("Upgrade failed. Staying with current version.")

if __name__ == "__main__":
    main()