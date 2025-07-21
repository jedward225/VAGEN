#!/usr/bin/env python3
"""
Final test of the get_map functionality after fixing the crash bug.
"""

import os
import sys
sys.path.append('/home/jiajunliu/VAGEN')

from vagen.env.spoc.env_config import SpocEnvConfig
from vagen.env.spoc.env import SpocEnv

def final_map_test():
    """Final test after fixing the string/dict bug."""
    
    SPOC_DATA_PATH = os.environ.get("SPOC_DATA_PATH", "/home/jiajunliu/spoc_data/fifteen") 
    os.environ["SPOC_DATA_PATH"] = SPOC_DATA_PATH
    
    output_dir = "./FINAL_MAP_TEST"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output: {output_dir}/")
    
    config = SpocEnvConfig(
        data_path=SPOC_DATA_PATH,
        task_type="FetchType", 
        chores_split="train"
    )
    
    env = SpocEnv(config)
    
    try:
        obs, info = env.reset(seed=999)
        target = env.episode_data.get('targetObjectType', 'unknown')
        print(f"🎯 Target object: {target}")
        
        # Save initial view
        if '<image>' in obs.get('multi_modal_data', {}):
            initial_img = obs['multi_modal_data']['<image>'][0]
            initial_img.save(f"{output_dir}/BEFORE_initial_view.png")
            print("💾 Initial view saved")
        
        print(f"\n🗺️  EXECUTING get_map (fixed version)...")
        
        # Test get_map - this should now work without crashing
        obs, reward, done, info = env.step("get_map")
        
        print(f"✅ Action executed successfully!")
        print(f"💰 Reward: {reward:.3f}")
        print(f"🔍 Action success: {info.get('last_action_success', False)}")
        
        # Check observation keys
        obs_keys = list(obs.get('multi_modal_data', {}).keys())
        print(f"📊 Available observation keys: {obs_keys}")
        
        # Save regular view after get_map
        if '<image>' in obs.get('multi_modal_data', {}):
            after_view = obs['multi_modal_data']['<image>'][0]
            after_view.save(f"{output_dir}/AFTER_get_map_regular_view.png")
            print("📸 Regular view after get_map saved")
        
        # THE MOMENT OF TRUTH: Check for top-down map
        map_placeholder = getattr(config, "map_placeholder", "<map>")
        if map_placeholder in obs.get('multi_modal_data', {}):
            map_img = obs['multi_modal_data'][map_placeholder][0]
            map_img.save(f"{output_dir}/🗺️_SUCCESS_TOPDOWN_MAP_🗺️.png")
            print(f"🎉🎉🎉 SUCCESS! TOP-DOWN MAP GENERATED! 🎉🎉🎉")
            print(f"🗺️  Saved as: 🗺️_SUCCESS_TOPDOWN_MAP_🗺️.png")
            print(f"📍 This should show:")
            print(f"   • Bird's-eye view of the room")
            print(f"   • Red markers for target objects ({target})")
            print(f"   • Agent position and path")
            print(f"   • Room layout from above")
        else:
            print(f"❌ Map still not found in observation")
            print(f"   Looking for key: '{map_placeholder}'")
            
        # Also check if map data exists in environment
        if hasattr(env, 'current_topdown_map') and env.current_topdown_map is not None:
            print(f"📊 Map data in env: shape={env.current_topdown_map.shape}")
            # Manual save as backup
            from vagen.env.utils.context_utils import convert_numpy_to_PIL
            manual_map = convert_numpy_to_PIL(env.current_topdown_map)
            manual_map.save(f"{output_dir}/BACKUP_map_from_env_data.png")
            print(f"🔧 Backup map saved from env data")
        else:
            print(f"❌ No map data in environment")
            
        return output_dir
        
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        env.close()

if __name__ == "__main__":
    print("🏁 FINAL MAP FUNCTIONALITY TEST")
    print("=" * 50)
    print("Testing after fixing the string/dict crash bug...")
    print()
    
    result_dir = final_map_test()
    
    if result_dir:
        print(f"\n" + "=" * 50)
        print(f"📁 CHECK RESULTS: {result_dir}/")
        print(f"🗺️  LOOK FOR: 🗺️_SUCCESS_TOPDOWN_MAP_🗺️.png")
        print(f"This should be a REAL bird's-eye view map!")
        print("=" * 50)
    else:
        print(f"\n❌ Test failed - check error above")