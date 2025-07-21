#!/usr/bin/env python3
"""
Test the fixed synchronized map functionality.
"""

import os
import sys
sys.path.append('/home/jiajunliu/VAGEN')

from vagen.env.spoc.env_config import SpocEnvConfig
from vagen.env.spoc.env import SpocEnv

def test_fixed_sync():
    """Test the fixed map synchronization."""
    
    SPOC_DATA_PATH = os.environ.get("SPOC_DATA_PATH", "/home/jiajunliu/spoc_data/fifteen") 
    os.environ["SPOC_DATA_PATH"] = SPOC_DATA_PATH
    
    output_dir = "./FIXED_SYNC_TEST"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output: {output_dir}/")
    
    config = SpocEnvConfig(
        data_path=SPOC_DATA_PATH,
        task_type="FetchType", 
        chores_split="train"
    )
    
    env = SpocEnv(config)
    
    try:
        obs, info = env.reset(seed=11111)
        target = env.episode_data.get('targetObjectType', 'unknown')
        scene_name = env.episode_data.get('scene', 'unknown')
        
        print(f"🎯 Target: {target}")
        print(f"🏠 Scene: {scene_name}")
        
        # Get scene info
        agent_pos = env.env.last_event.metadata["agent"]["position"]
        print(f"📍 Agent at: ({agent_pos['x']:.2f}, {agent_pos['z']:.2f})")
        
        # Save first-person view
        if '<image>' in obs.get('multi_modal_data', {}):
            first_person = obs['multi_modal_data']['<image>'][0]
            first_person.save(f"{output_dir}/FIRST_PERSON_{scene_name}.png")
            print(f"📸 First-person view saved for scene {scene_name}")
        
        print(f"\n🗺️  Generating map for scene {scene_name}...")
        
        # Generate map
        obs, reward, done, info = env.step("get_map")
        
        print(f"✅ Map generation result: {info.get('last_action_success', False)}")
        
        # Save the map
        map_placeholder = getattr(config, "map_placeholder", "<map>")
        if map_placeholder in obs.get('multi_modal_data', {}):
            map_img = obs['multi_modal_data'][map_placeholder][0]
            map_img.save(f"{output_dir}/MAP_{scene_name}.png")
            
            print(f"🎉 SUCCESS!")
            print(f"🗺️  Map saved for scene: {scene_name}")
            print(f"📍 Map should show agent at position ({agent_pos['x']:.2f}, {agent_pos['z']:.2f})")
            print(f"🎯 Red markers should show {target} locations")
            
            return True
        else:
            print("❌ Map still not generated")
            return False
            
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        env.close()

if __name__ == "__main__":
    print("🔧 FIXED SYNCHRONIZATION TEST")
    print("=" * 50)
    
    success = test_fixed_sync()
    
    print(f"\n" + "=" * 50)
    if success:
        print("✅ Fixed synchronization working!")
        print("🗺️  Check FIXED_SYNC_TEST/ for results")
    else:
        print("❌ Still having issues")
    print("=" * 50)