#!/usr/bin/env python3
"""
FINAL WORKING MAP TEST - Fixed RGBA to RGB conversion issue.
"""

import os
import sys
sys.path.append('/home/jiajunliu/VAGEN')

from vagen.env.spoc.env_config import SpocEnvConfig
from vagen.env.spoc.env import SpocEnv

def final_working_test():
    """Final test with RGBA conversion fix."""
    
    SPOC_DATA_PATH = os.environ.get("SPOC_DATA_PATH", "/home/jiajunliu/spoc_data/fifteen") 
    os.environ["SPOC_DATA_PATH"] = SPOC_DATA_PATH
    
    output_dir = "./FINAL_WORKING_MAP_TEST"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}/")
    
    config = SpocEnvConfig(
        data_path=SPOC_DATA_PATH,
        task_type="FetchType", 
        chores_split="train"
    )
    
    env = SpocEnv(config)
    
    try:
        obs, info = env.reset(seed=12345)
        target = env.episode_data.get('targetObjectType', 'unknown')
        print(f"🎯 Target: {target}")
        
        # Save initial view
        if '<image>' in obs.get('multi_modal_data', {}):
            initial_img = obs['multi_modal_data']['<image>'][0]
            initial_img.save(f"{output_dir}/01_BEFORE_regular_view.png")
            print("💾 Initial view saved")
        
        print(f"\n🚀 EXECUTING get_map (with RGBA fix)...")
        
        # Execute get_map action
        obs, reward, done, info = env.step("get_map")
        
        print(f"✅ Execution complete!")
        print(f"💰 Reward: {reward:.3f}")
        print(f"🔍 Action success: {info.get('last_action_success', False)}")
        
        # Check what's in observation now
        obs_keys = list(obs.get('multi_modal_data', {}).keys())
        print(f"📊 Observation keys: {obs_keys}")
        
        # Save regular view after get_map
        if '<image>' in obs.get('multi_modal_data', {}):
            after_img = obs['multi_modal_data']['<image>'][0]
            after_img.save(f"{output_dir}/02_AFTER_regular_view.png")
            print("📸 Regular view after get_map saved")
        
        # THE ULTIMATE TEST: Check for the map
        map_placeholder = getattr(config, "map_placeholder", "<map>")
        print(f"🔍 Looking for map with placeholder: '{map_placeholder}'")
        
        if map_placeholder in obs.get('multi_modal_data', {}):
            map_img = obs['multi_modal_data'][map_placeholder][0]
            map_filename = f"{output_dir}/🎉_SUCCESS_TOPDOWN_MAP_🎉.png"
            map_img.save(map_filename)
            
            print(f"\n" + "🎉" * 50)
            print(f"🎉 FINALLY! SUCCESS! BIRD'S-EYE VIEW MAP GENERATED!")
            print(f"🎉" * 50)
            print(f"🗺️  File: 🎉_SUCCESS_TOPDOWN_MAP_🎉.png")
            print(f"📍 This shows:")
            print(f"   • Top-down view of the room layout")
            print(f"   • Red markers for target objects ({target})")  
            print(f"   • Agent's current position")
            print(f"   • Strategic navigation information")
            print(f"🎉" * 50)
            
            # Also save environment map data as backup
            if hasattr(env, 'current_topdown_map') and env.current_topdown_map is not None:
                print(f"📊 Confirmed: Map data in env with shape {env.current_topdown_map.shape}")
            
            return True
        else:
            print(f"❌ Map still not found in observation")
            print(f"   Available keys: {obs_keys}")
            print(f"   Expected key: '{map_placeholder}'")
            return False
            
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        env.close()

if __name__ == "__main__":
    print("🏁 FINAL WORKING MAP TEST")
    print("=" * 60)
    print("Fixed the RGBA → RGB conversion issue")
    print("This should FINALLY generate the bird's-eye view map!")
    print("=" * 60)
    print()
    
    success = final_working_test()
    
    print(f"\n" + "=" * 60)
    if success:
        print("✅ SUCCESS! Map functionality is now working!")
        print("🗺️  Check FINAL_WORKING_MAP_TEST/ for the bird's-eye view!")
    else:
        print("❌ Still failed - something else is wrong")
    print("=" * 60)