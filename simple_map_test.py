#!/usr/bin/env python3
"""
Simple working test for SPOC map functionality after fixing parsing issues.
"""

import os
import sys
sys.path.append('/home/jiajunliu/VAGEN')

from vagen.env.spoc.env_config import SpocEnvConfig
from vagen.env.spoc.env import SpocEnv

def test_get_map_action():
    """Test the get_map action with fixed parsing."""
    
    # Set up environment
    SPOC_DATA_PATH = os.environ.get("SPOC_DATA_PATH", "/home/jiajunliu/spoc_data/fifteen")
    os.environ["SPOC_DATA_PATH"] = SPOC_DATA_PATH
    print(f"📁 Using SPOC data: {SPOC_DATA_PATH}")
    
    # Create output directory
    output_dir = "./simple_map_test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create config
    config = SpocEnvConfig(
        data_path=SPOC_DATA_PATH,
        task_type="FetchType",
        chores_split="train"
    )
    
    # Initialize environment
    print("🚀 Initializing SPOC Environment...")
    env = SpocEnv(config)
    
    try:
        # Reset environment
        obs, info = env.reset(seed=456)
        print(f"✅ Environment reset successful")
        print(f"🎯 Target: {env.episode_data.get('targetObjectType', 'unknown')}")
        
        # Save initial view
        if '<image>' in obs.get('multi_modal_data', {}):
            initial_img = obs['multi_modal_data']['<image>'][0]
            initial_img.save(f"{output_dir}/00_initial_view.png")
            print("💾 Saved initial view")
        
        # Test simple get_map action (should work with fixed parsing)
        print(f"\n🗺️  Testing get_map action...")
        
        obs, reward, done, info = env.step("get_map")
        
        print(f"📊 Action executed: get_map")
        print(f"✅ Action Success: {info.get('last_action_success', False)}")
        print(f"💰 Reward: {reward:.3f}")
        
        # Check observation keys
        obs_keys = list(obs.get('multi_modal_data', {}).keys())
        print(f"🔍 Available observation keys: {obs_keys}")
        
        # Check for map in observation
        map_placeholder = getattr(config, "map_placeholder", "<map>")
        if map_placeholder in obs.get('multi_modal_data', {}):
            map_img = obs['multi_modal_data'][map_placeholder][0]
            map_img.save(f"{output_dir}/01_SUCCESS_map_image.png")
            print(f"🎉 SUCCESS! Map image found and saved!")
            print(f"🗺️  File: 01_SUCCESS_map_image.png")
        else:
            print(f"⚠️  No map in observation with key '{map_placeholder}'")
            
        # Check environment map storage
        if hasattr(env, 'current_topdown_map') and env.current_topdown_map is not None:
            print(f"📊 Map stored in env: shape={env.current_topdown_map.shape}")
            # Manually convert and save
            from vagen.env.utils.context_utils import convert_numpy_to_PIL
            manual_map = convert_numpy_to_PIL(env.current_topdown_map)
            manual_map.save(f"{output_dir}/01_manual_map_from_env.png")
            print(f"🗺️  Also saved manual extract: 01_manual_map_from_env.png")
        
        # Test a few more actions
        test_actions = ["moveahead", "rotateright", "get_map"]
        
        for i, action in enumerate(test_actions, 2):
            print(f"\n📍 Step {i}: {action}")
            obs, reward, done, info = env.step(action)
            
            print(f"✅ Success: {info.get('last_action_success', False)}")
            print(f"💰 Reward: {reward:.3f}")
            
            # Save regular view
            if '<image>' in obs.get('multi_modal_data', {}):
                img = obs['multi_modal_data']['<image>'][0]
                img.save(f"{output_dir}/{i:02d}_view_{action}.png")
                print(f"📸 View saved: {i:02d}_view_{action}.png")
            
            # Save map if it's a map action
            if action == "get_map" and map_placeholder in obs.get('multi_modal_data', {}):
                map_img = obs['multi_modal_data'][map_placeholder][0]
                map_img.save(f"{output_dir}/{i:02d}_MAP_{action}.png")
                print(f"🗺️  Map saved: {i:02d}_MAP_{action}.png")
        
        print(f"\n🎉 Test completed! Check {output_dir}/ for results")
        return output_dir
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        env.close()

if __name__ == "__main__":
    print("🧪 Simple Map Functionality Test")
    print("================================")
    print("Testing get_map action after parsing fixes...")
    print()
    
    result_dir = test_get_map_action()
    
    if result_dir:
        print(f"\n✅ Test completed successfully!")
        print(f"📁 Results: {result_dir}/")
        print(f"🗺️  Look for files containing 'MAP' in the name")
    else:
        print(f"\n❌ Test failed - check error messages above")