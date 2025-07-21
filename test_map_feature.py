#!/usr/bin/env python3
"""
Enhanced test script for SPOC environment map functionality.
This script demonstrates the bird's-eye view map by:
1. Initializing the environment
2. Taking navigation actions and generating maps
3. Saving and displaying the map images
"""

import os
import sys
import time
sys.path.append('/home/jiajunliu/VAGEN')

from vagen.env.spoc.env_config import SpocEnvConfig
from vagen.env.spoc.env import SpocEnv

def test_map_functionality():
    """Test the map generation and save actual map images."""
    
    # Set up environment
    SPOC_DATA_PATH = os.environ.get("SPOC_DATA_PATH") 
    if not SPOC_DATA_PATH:
        print("Setting default SPOC_DATA_PATH...")
        SPOC_DATA_PATH = "/home/jiajunliu/spoc_data/fifteen"
        os.environ["SPOC_DATA_PATH"] = SPOC_DATA_PATH
        print(f"Using: {SPOC_DATA_PATH}")
    
    # Create output directory for images
    output_dir = "./map_test_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Create config
    config = SpocEnvConfig(
        data_path=SPOC_DATA_PATH,
        task_type="FetchType",
        chores_split="train"
    )
    
    # Initialize environment
    print("🚀 Initializing SPOC Environment...")
    env = SpocEnv(config)
    
    # Reset to get a random episode
    obs, info = env.reset(seed=42)  # Fixed seed for reproducibility
    print(f"\n📋 Episode Task: {info.get('instruction', 'Unknown task')}")
    print(f"🎯 Target Object: {env.episode_data.get('targetObjectType', 'Unknown')}")
    
    # Save initial regular view
    if '<image>' in obs.get('multi_modal_data', {}):
        initial_img = obs['multi_modal_data']['<image>'][0]
        initial_img.save(f"{output_dir}/step_0_initial_view.png")
        print("💾 Saved initial view image")
    
    # Define test sequence with map generation
    test_sequence = [
        ("get_map", "Generate initial bird's-eye view map"),
        ("moveahead", "Move forward to explore"),
        ("rotateright", "Rotate right 30 degrees"),
        ("get_map", "Generate map after rotation"),
        ("moveahead", "Move forward again"),
        ("moveahead", "Continue exploring"),
        ("get_map", "Generate map after exploration"),
        ("rotateleft", "Rotate left"),
        ("move_arm_out", "Extend arm for potential pickup"),
        ("get_map", "Final map with arm extended")
    ]
    
    print(f"\n🎮 Executing test sequence ({len(test_sequence)} actions):")
    print("=" * 60)
    
    for step, (action_name, description) in enumerate(test_sequence, 1):
        print(f"\n📍 Step {step}: {description}")
        print(f"🎯 Action: {action_name}")
        
        # Format action for the environment (simulating LLM response)
        action_str = f"<think>{description}</think><answer>{action_name}</answer>"
        
        # Execute action
        obs, reward, done, info = env.step(action_str)
        
        # Print status
        print(f"✅ Action Success: {info.get('last_action_success', False)}")
        print(f"💰 Reward: {reward:.3f}")
        print(f"🏆 Task Success: {info.get('task_success', False)}")
        
        # Save regular camera view
        if '<image>' in obs.get('multi_modal_data', {}):
            img = obs['multi_modal_data']['<image>'][0]
            img.save(f"{output_dir}/step_{step:02d}_view_{action_name}.png")
            print(f"📸 Saved camera view: step_{step:02d}_view_{action_name}.png")
        
        # Save map image if available
        map_placeholder = getattr(config, "map_placeholder", "<map>")
        if map_placeholder in obs.get('multi_modal_data', {}):
            map_img = obs['multi_modal_data'][map_placeholder][0]
            map_img.save(f"{output_dir}/step_{step:02d}_MAP_{action_name}.png")
            print(f"🗺️  SAVED MAP IMAGE: step_{step:02d}_MAP_{action_name}.png")
            print(f"    Map shows: Room layout, objects (red markers), agent position")
        elif action_name == "get_map":
            print(f"⚠️  Map action executed but no map image found in observation")
            # Check if map is stored in environment
            if hasattr(env, 'current_topdown_map') and env.current_topdown_map is not None:
                print(f"📊 Map data exists in env: shape={env.current_topdown_map.shape}")
        
        # Display available observation keys
        obs_keys = list(obs.get('multi_modal_data', {}).keys())
        print(f"🔍 Available observation keys: {obs_keys}")
        
        print("-" * 40)
        
        if done or info.get('task_success'):
            print("🏆 Episode completed - task successful!")
            break
            
        time.sleep(0.3)  # Small delay for readability
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("📈 TEST SUMMARY:")
    print("=" * 60)
    print(f"📁 Images saved to: {output_dir}/")
    print(f"🖼️  Camera views: step_XX_view_ACTION.png")
    print(f"🗺️  Maps: step_XX_MAP_ACTION.png")
    print(f"\nMap images contain:")
    print(f"  • 🔴 Red markers: Target objects")
    print(f"  • 📍 Agent position and movement path")
    print(f"  • 🏠 Room layout, walls, and furniture")
    print(f"  • 🎯 Strategic view for navigation planning")
    
    env.close()
    print("\n✅ Map functionality test completed!")
    return output_dir

if __name__ == "__main__":
    try:
        output_dir = test_map_functionality()
        print(f"\n🎉 Success! Check {output_dir}/ for generated map images!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()