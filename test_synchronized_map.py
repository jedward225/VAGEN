#!/usr/bin/env python3
"""
Test the synchronized map functionality - map should show same scene as first-person view.
"""

import os
import sys
sys.path.append('/home/jiajunliu/VAGEN')

from vagen.env.spoc.env_config import SpocEnvConfig
from vagen.env.spoc.env import SpocEnv

def test_synchronized_map():
    """Test that the map shows the same scene as the first-person view."""
    
    SPOC_DATA_PATH = os.environ.get("SPOC_DATA_PATH", "/home/jiajunliu/spoc_data/fifteen") 
    os.environ["SPOC_DATA_PATH"] = SPOC_DATA_PATH
    
    output_dir = "./SYNCHRONIZED_MAP_TEST"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output: {output_dir}/")
    
    config = SpocEnvConfig(
        data_path=SPOC_DATA_PATH,
        task_type="FetchType", 
        chores_split="train"
    )
    
    env = SpocEnv(config)
    
    try:
        obs, info = env.reset(seed=54321)  # Different seed to test new scene
        target = env.episode_data.get('targetObjectType', 'unknown')
        scene_name = env.episode_data.get('scene', 'unknown')
        print(f"🎯 Target: {target}")
        print(f"🏠 Scene: {scene_name}")
        
        # Get agent position info
        agent_pos = env.env.last_event.metadata["agent"]["position"]
        print(f"📍 Agent position: x={agent_pos['x']:.2f}, z={agent_pos['z']:.2f}")
        
        # Save first-person view BEFORE map
        if '<image>' in obs.get('multi_modal_data', {}):
            first_person = obs['multi_modal_data']['<image>'][0]
            first_person.save(f"{output_dir}/01_FIRST_PERSON_view.png")
            print("📸 First-person view saved")
        
        print(f"\n🗺️  Generating synchronized map...")
        
        # Generate map using fixed synchronization
        obs, reward, done, info = env.step("get_map")
        
        print(f"✅ Map generation complete")
        print(f"💰 Reward: {reward:.3f}")
        
        # Save first-person view AFTER map (should be same scene)
        if '<image>' in obs.get('multi_modal_data', {}):
            after_first_person = obs['multi_modal_data']['<image>'][0]
            after_first_person.save(f"{output_dir}/02_FIRST_PERSON_after_map.png")
            print("📸 First-person after map saved")
        
        # Save the synchronized map
        map_placeholder = getattr(config, "map_placeholder", "<map>")
        if map_placeholder in obs.get('multi_modal_data', {}):
            map_img = obs['multi_modal_data'][map_placeholder][0]
            map_img.save(f"{output_dir}/03_SYNCHRONIZED_MAP.png")
            print("🗺️  Synchronized map saved!")
            
            print(f"\n🔍 VERIFICATION:")
            print(f"   • First-person view: Shows scene '{scene_name}'")
            print(f"   • Map view: Should show SAME scene '{scene_name}' from above")
            print(f"   • Agent position: ({agent_pos['x']:.2f}, {agent_pos['z']:.2f})")
            print(f"   • Target objects: {target} should be visible as red markers")
            
            return True
        else:
            print("❌ Map not found in observation")
            return False
            
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        env.close()

if __name__ == "__main__":
    print("🔄 SYNCHRONIZED MAP TEST")
    print("=" * 60)
    print("Testing map synchronization with first-person view")
    print("Map should show SAME SCENE as first-person camera")
    print("=" * 60)
    print()
    
    success = test_synchronized_map()
    
    print(f"\n" + "=" * 60)
    if success:
        print("✅ Synchronization test complete!")
        print("🔍 Compare images:")
        print("   01_FIRST_PERSON_view.png (before map)")
        print("   02_FIRST_PERSON_after_map.png (after map)")  
        print("   03_SYNCHRONIZED_MAP.png (bird's-eye view)")
        print("👀 The map should show the same room/environment!")
    else:
        print("❌ Synchronization test failed")
    print("=" * 60)