#!/usr/bin/env python3
"""
Find and test a REAL indoor scene with furniture, not outdoor scenes.
"""

import os
import sys
import random
sys.path.append('/home/jiajunliu/VAGEN')

from vagen.env.spoc.env_config import SpocEnvConfig
from vagen.env.spoc.env import SpocEnv

def find_real_indoor_scene():
    """Search through episodes to find actual indoor scenes with furniture."""
    
    SPOC_DATA_PATH = os.environ.get("SPOC_DATA_PATH", "/home/jiajunliu/spoc_data/fifteen") 
    os.environ["SPOC_DATA_PATH"] = SPOC_DATA_PATH
    
    output_dir = "./REAL_INDOOR_MAP"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output: {output_dir}/")
    
    config = SpocEnvConfig(
        data_path=SPOC_DATA_PATH,
        task_type="FetchType", 
        chores_split="train"
    )
    
    env = SpocEnv(config)
    
    print("🔍 Searching for real indoor scenes with furniture...")
    print("Looking for scenes with kitchen/living room/bedroom indicators...\n")
    
    # Try a wider range of episodes to find indoor scenes
    # SPOC has specific scene types - let's look for ones likely to be indoors
    attempts = 0
    max_attempts = 50
    
    # Try random episodes
    while attempts < max_attempts:
        attempts += 1
        
        try:
            # Random episode
            episode_idx = random.randint(0, len(env.dataset) - 1)
            obs, info = env.reset(seed=episode_idx)
            
            scene_name = env.episode_data.get('scene', 'unknown')
            target = env.episode_data.get('targetObjectType', 'unknown')
            
            # Check if we have objects visible that indicate indoor scene
            objects = env.env.last_event.metadata.get("objects", [])
            indoor_indicators = ['Table', 'Chair', 'Sofa', 'Bed', 'Cabinet', 'Fridge', 
                               'Microwave', 'Sink', 'Television', 'Desk', 'Lamp']
            
            indoor_objects_found = []
            for obj in objects:
                obj_type = obj.get("objectType", "")
                if any(indicator in obj_type for indicator in indoor_indicators):
                    indoor_objects_found.append(obj_type)
            
            if indoor_objects_found:
                print(f"✅ Found indoor scene! Episode {episode_idx}")
                print(f"🏠 Scene: {scene_name}")
                print(f"🪑 Indoor objects detected: {indoor_objects_found[:5]}")
                print(f"🎯 Target: {target}")
                
                agent_pos = env.env.last_event.metadata["agent"]["position"]
                print(f"📍 Agent at: ({agent_pos['x']:.2f}, {agent_pos['z']:.2f})")
                
                # Save first-person view
                if '<image>' in obs.get('multi_modal_data', {}):
                    fp_img = obs['multi_modal_data']['<image>'][0]
                    fp_img.save(f"{output_dir}/1_INDOOR_FirstPerson_{scene_name}.png")
                    print("📸 Saved first-person indoor view")
                
                # Move and rotate to see more of the room
                print("\n🚶 Moving around to see the room...")
                env.step("moveahead")
                env.step("moveahead")
                obs, _, _, _ = env.step("rotateright")
                
                if '<image>' in obs.get('multi_modal_data', {}):
                    rotated_img = obs['multi_modal_data']['<image>'][0]
                    rotated_img.save(f"{output_dir}/2_INDOOR_Rotated_{scene_name}.png")
                    print("📸 Saved rotated indoor view")
                
                # Generate map
                print("\n🗺️  Generating top-down map of indoor scene...")
                obs, reward, done, info = env.step("get_map")
                
                # Save map
                map_placeholder = getattr(config, "map_placeholder", "<map>")
                if map_placeholder in obs.get('multi_modal_data', {}):
                    map_img = obs['multi_modal_data'][map_placeholder][0]
                    map_img.save(f"{output_dir}/3_INDOOR_MAP_TopDown_{scene_name}.png")
                    
                    print("\n🎉 SUCCESS! Indoor map generated!")
                    print("🗺️  The map should clearly show:")
                    print("   • Room layout from ABOVE")
                    print("   • Furniture tops (tables, chairs, etc.)")
                    print("   • Floor patterns")
                    print("   • NOT a side view of furniture")
                    
                    # List what should be visible
                    print(f"\n📋 Objects in this indoor scene:")
                    for i, obj in enumerate(indoor_objects_found[:10]):
                        print(f"   {i+1}. {obj}")
                    
                    return True
                else:
                    print("❌ Map generation failed")
            else:
                if attempts % 10 == 0:
                    print(f"Attempt {attempts}: Scene {scene_name} appears to be outdoor, continuing search...")
                    
        except Exception as e:
            print(f"Error in attempt {attempts}: {e}")
            continue
    
    print(f"\n❌ Could not find indoor scene after {max_attempts} attempts")
    print("The dataset might have mostly outdoor scenes")
    return False

if __name__ == "__main__":
    print("🏠 FINDING REAL INDOOR SCENES")
    print("=" * 60)
    print("Searching for scenes with furniture to demonstrate map view")
    print("=" * 60)
    
    success = find_real_indoor_scene()
    
    print(f"\n" + "=" * 60)
    if success:
        print("✅ Found and mapped an indoor scene!")
        print("🗺️  Check REAL_INDOOR_MAP/ folder")
        print("\n🔍 Compare the images:")
        print("1. First-person: See furniture from the SIDE")
        print("2. Map: See furniture from ABOVE") 
        print("\nThe difference should be obvious!")
    else:
        print("❌ Could not find suitable indoor scene")
        print("The SPOC dataset might be mostly outdoor scenes")
    print("=" * 60)