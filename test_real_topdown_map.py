#!/usr/bin/env python3
"""
Test REAL top-down map generation - camera positioned HIGH above scene.
"""

import os
import sys
sys.path.append('/home/jiajunliu/VAGEN')

from vagen.env.spoc.env_config import SpocEnvConfig
from vagen.env.spoc.env import SpocEnv

def test_real_topdown():
    """Test the REAL top-down map with high camera."""
    
    SPOC_DATA_PATH = os.environ.get("SPOC_DATA_PATH", "/home/jiajunliu/spoc_data/fifteen") 
    os.environ["SPOC_DATA_PATH"] = SPOC_DATA_PATH
    
    output_dir = "./REAL_TOPDOWN_TEST"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output: {output_dir}/")
    
    config = SpocEnvConfig(
        data_path=SPOC_DATA_PATH,
        task_type="FetchType", 
        chores_split="train"
    )
    
    env = SpocEnv(config)
    
    try:
        # Use a specific seed
        obs, info = env.reset(seed=98765)
        scene = env.episode_data.get('scene', 'unknown')
        target = env.episode_data.get('targetObjectType', 'unknown')
        
        print(f"🏠 Scene: {scene}")
        print(f"🎯 Target: {target}")
        
        agent_pos = env.env.last_event.metadata["agent"]["position"]
        print(f"📍 Agent position: ({agent_pos['x']:.2f}, {agent_pos['y']:.2f}, {agent_pos['z']:.2f})")
        
        # Save first-person FORWARD view
        if '<image>' in obs.get('multi_modal_data', {}):
            fp_img = obs['multi_modal_data']['<image>'][0]
            fp_img.save(f"{output_dir}/1_FirstPerson_FORWARD.png")
            print("📸 Saved: First-person looking FORWARD")
        
        # Look DOWN from first-person
        obs, _, _, _ = env.step("lookdown")
        if '<image>' in obs.get('multi_modal_data', {}):
            look_down_img = obs['multi_modal_data']['<image>'][0]
            look_down_img.save(f"{output_dir}/2_FirstPerson_LOOKING_DOWN.png")
            print("📸 Saved: First-person looking DOWN (this is NOT a map!)")
        
        # Look back up
        env.step("lookup")
        
        # Generate REAL top-down map
        print(f"\n🗺️  Generating REAL top-down map from 8m above...")
        obs, reward, done, info = env.step("get_map")
        
        print(f"✅ Map generation complete")
        
        # Save the REAL top-down map
        map_placeholder = getattr(config, "map_placeholder", "<map>")
        if map_placeholder in obs.get('multi_modal_data', {}):
            map_img = obs['multi_modal_data'][map_placeholder][0]
            map_img.save(f"{output_dir}/3_REAL_TOPDOWN_MAP_FROM_8M_HIGH.png")
            
            print(f"\n🎉 SUCCESS! REAL top-down map saved!")
            print(f"🗺️  This map is taken from 8 meters ABOVE the agent")
            print(f"📐 Using orthographic projection (no perspective distortion)")
            print(f"👁️ Camera pointing straight DOWN (90° angle)")
            
            print(f"\n🔍 COMPARE THE 3 IMAGES:")
            print(f"1. First-person FORWARD - normal eye-level view")
            print(f"2. First-person LOOKING DOWN - agent tilting head down") 
            print(f"3. REAL TOP-DOWN MAP - camera 8m above, looking straight down")
            print(f"\nThe map should show a MUCH WIDER area from directly above!")
            
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
    print("🚁 REAL TOP-DOWN MAP TEST")
    print("=" * 60)
    print("Testing with camera positioned 8 meters above agent")
    print("This should produce a TRUE bird's-eye view map")
    print("=" * 60)
    print()
    
    success = test_real_topdown()
    
    print(f"\n" + "=" * 60)
    if success:
        print("✅ Real top-down map test complete!")
        print("📂 Check REAL_TOPDOWN_TEST/ folder")
        print("\n🔍 The difference should be obvious:")
        print("   Image 2: Agent looking down = limited floor view")
        print("   Image 3: Real map = wide area from high above!")
    else:
        print("❌ Real top-down map test failed")
    print("=" * 60)