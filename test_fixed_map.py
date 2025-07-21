#!/usr/bin/env python3
"""
Test the fixed get_map functionality.
"""

import os
import sys
sys.path.append('/home/jiajunliu/VAGEN')

from vagen.env.spoc.env_config import SpocEnvConfig
from vagen.env.spoc.env import SpocEnv

def test_fixed_get_map():
    """Test get_map after fixing the execution issue."""
    
    SPOC_DATA_PATH = os.environ.get("SPOC_DATA_PATH", "/home/jiajunliu/spoc_data/fifteen") 
    os.environ["SPOC_DATA_PATH"] = SPOC_DATA_PATH
    
    output_dir = "./test_fixed_map_output"
    os.makedirs(output_dir, exist_ok=True)
    
    config = SpocEnvConfig(
        data_path=SPOC_DATA_PATH,
        task_type="FetchType", 
        chores_split="train"
    )
    
    env = SpocEnv(config)
    
    try:
        obs, info = env.reset(seed=789)
        print(f"🎯 Target: {env.episode_data.get('targetObjectType', 'unknown')}")
        
        # Save initial view
        if '<image>' in obs.get('multi_modal_data', {}):
            initial_img = obs['multi_modal_data']['<image>'][0]
            initial_img.save(f"{output_dir}/01_initial_view.png")
            print("💾 Initial view saved")
        
        print(f"\n🗺️  Testing get_map action (should now execute properly)...")
        
        # Test get_map
        obs, reward, done, info = env.step("get_map")
        
        print(f"✅ Action Success: {info.get('last_action_success', False)}")
        print(f"💰 Reward: {reward:.3f}")
        
        # Check if map appears in observation
        obs_keys = list(obs.get('multi_modal_data', {}).keys())
        print(f"🔍 Observation keys after get_map: {obs_keys}")
        
        # Save view after get_map
        if '<image>' in obs.get('multi_modal_data', {}):
            view_img = obs['multi_modal_data']['<image>'][0]
            view_img.save(f"{output_dir}/02_view_after_get_map.png")
            print("📸 View after get_map saved")
        
        # Save map if present
        map_placeholder = getattr(config, "map_placeholder", "<map>")
        if map_placeholder in obs.get('multi_modal_data', {}):
            map_img = obs['multi_modal_data'][map_placeholder][0]
            map_img.save(f"{output_dir}/02_SUCCESS_TOPDOWN_MAP.png")
            print(f"🎉 SUCCESS! Top-down map saved!")
            print(f"🗺️  This should show bird's-eye view with red target markers")
        else:
            print(f"❌ Still no map in observation")
            
        # Check environment map storage
        if hasattr(env, 'current_topdown_map') and env.current_topdown_map is not None:
            print(f"📊 Map in environment: shape={env.current_topdown_map.shape}")
        else:
            print(f"❌ No map stored in environment either")
            
        return output_dir
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        env.close()

if __name__ == "__main__":
    print("🔧 Testing Fixed Map Functionality")
    print("==================================")
    
    result_dir = test_fixed_get_map()
    if result_dir:
        print(f"\n📁 Check: {result_dir}/")
        print(f"🗺️  Look for: 02_SUCCESS_TOPDOWN_MAP.png")
        print(f"This should be a bird's-eye view, NOT a regular camera view!")