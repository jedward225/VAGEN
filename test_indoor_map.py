#!/usr/bin/env python3
"""
Test map functionality with an indoor scene that has clear furniture and room layout.
"""

import os
import sys
sys.path.append('/home/jiajunliu/VAGEN')

from vagen.env.spoc.env_config import SpocEnvConfig
from vagen.env.spoc.env import SpocEnv

def test_indoor_map():
    """Test map generation with an indoor scene for clear top-down view."""
    
    SPOC_DATA_PATH = os.environ.get("SPOC_DATA_PATH", "/home/jiajunliu/spoc_data/fifteen") 
    os.environ["SPOC_DATA_PATH"] = SPOC_DATA_PATH
    
    output_dir = "./INDOOR_MAP_TEST"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output: {output_dir}/")
    
    config = SpocEnvConfig(
        data_path=SPOC_DATA_PATH,
        task_type="FetchType", 
        chores_split="train"
    )
    
    env = SpocEnv(config)
    
    # Try multiple seeds to find a good indoor scene
    test_seeds = [777, 888, 999, 1234, 5678, 2468, 3579, 4680, 1357, 2222]
    
    for seed in test_seeds:
        try:
            obs, info = env.reset(seed=seed)
            scene_name = env.episode_data.get('scene', 'unknown')
            
            # Skip outdoor scenes (they often have "Floor" in the name without "Plan")
            if "Floor" not in scene_name or "Plan" not in scene_name:
                print(f"Skipping non-floor plan scene: {scene_name}")
                continue
                
            target = env.episode_data.get('targetObjectType', 'unknown')
            print(f"\n🏠 Testing scene: {scene_name}")
            print(f"🎯 Target: {target}")
            
            agent_pos = env.env.last_event.metadata["agent"]["position"]
            print(f"📍 Agent position: ({agent_pos['x']:.2f}, {agent_pos['z']:.2f})")
            
            # Save first-person view
            if '<image>' in obs.get('multi_modal_data', {}):
                first_person = obs['multi_modal_data']['<image>'][0]
                filename = f"{output_dir}/1_FIRST_PERSON_{scene_name}_seed{seed}.png"
                first_person.save(filename)
                print(f"📸 First-person view saved: {filename}")
            
            # Take a few actions to get inside if we're at a doorway
            actions_to_try = ["moveahead", "moveahead", "rotateright", "moveahead"]
            for i, action in enumerate(actions_to_try):
                obs, _, _, _ = env.step(action)
                if i == 2:  # Save view after rotating
                    if '<image>' in obs.get('multi_modal_data', {}):
                        rotated_view = obs['multi_modal_data']['<image>'][0]
                        rotated_filename = f"{output_dir}/2_ROTATED_VIEW_{scene_name}_seed{seed}.png"
                        rotated_view.save(rotated_filename)
                        print(f"📸 Rotated view saved: {rotated_filename}")
            
            # Generate map
            print(f"🗺️  Generating map for {scene_name}...")
            obs, reward, done, info = env.step("get_map")
            
            # Save the map
            map_placeholder = getattr(config, "map_placeholder", "<map>")
            if map_placeholder in obs.get('multi_modal_data', {}):
                map_img = obs['multi_modal_data'][map_placeholder][0]
                map_filename = f"{output_dir}/3_MAP_TOP_DOWN_{scene_name}_seed{seed}.png"
                map_img.save(map_filename)
                
                print(f"🎉 SUCCESS! Map saved: {map_filename}")
                print(f"🗺️  This should show:")
                print(f"   • Room layout from above")
                print(f"   • Furniture (tables, chairs, beds, etc.)")
                print(f"   • Walls and doorways")
                print(f"   • Agent position and target objects ({target})")
                
                # Save current view after map
                if '<image>' in obs.get('multi_modal_data', {}):
                    after_map_view = obs['multi_modal_data']['<image>'][0]
                    after_filename = f"{output_dir}/4_AFTER_MAP_VIEW_{scene_name}_seed{seed}.png"
                    after_map_view.save(after_filename)
                    print(f"📸 View after map saved: {after_filename}")
                
                print(f"\n✅ Indoor scene test complete for {scene_name}")
                print(f"Check the map to see furniture and room layout from above!")
                return True
                
            else:
                print(f"❌ Map generation failed for {scene_name}")
                
        except Exception as e:
            print(f"Error with seed {seed}: {e}")
            continue
    
    print("\n❌ Could not find a suitable indoor scene")
    return False
    
    env.close()

if __name__ == "__main__":
    print("🏠 INDOOR SCENE MAP TEST")
    print("=" * 60)
    print("Testing map with indoor scenes that have clear furniture")
    print("This will make it obvious the map is a top-down view")
    print("=" * 60)
    
    success = test_indoor_map()
    
    print(f"\n" + "=" * 60)
    if success:
        print("✅ Indoor map test successful!")
        print("🗺️  Check INDOOR_MAP_TEST/ folder")
        print("Compare:")
        print("  1. First-person view (eye level)")
        print("  2. Rotated view (different angle)")
        print("  3. MAP - should clearly show room from ABOVE")
        print("  4. After map view (back to eye level)")
    else:
        print("❌ Indoor map test failed")
    print("=" * 60)