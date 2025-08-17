#!/usr/bin/env python3
"""
Test the final SPOC map fix
"""

import os
import sys

# Set environment variables BEFORE importing
os.environ["OBJAVERSE_HOUSES_DIR"] = "/home/jiajunliu/objaverse_houses"
os.environ["OBJAVERSE_DATA_DIR"] = "/home/jiajunliu/objaverse_data"

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append('/home/jiajunliu/VAGEN')

from vagen.env.spoc.env import SpocEnv
from vagen.env.spoc.env_config import SpocEnvConfig

def main():
    """Test the fixed SPOC environment."""
    
    SPOC_DATA_PATH = "/home/jiajunliu/spoc_data/all"
    
    print("="*60)
    print("SPOC Final Map Fix Test")
    print("="*60)
    
    # Create environment
    config = SpocEnvConfig(
        data_path=SPOC_DATA_PATH,
        task_type="FetchType",
        chores_split="val",
        map_size=512,
        prompt_format="free_think",
        max_actions_per_step=1,
        action_sep=","
    )
    
    try:
        print("\n[INFO] Initializing SPOC environment...")
        env = SpocEnv(config)
        print(f"[SUCCESS] Environment initialized with {env.number_of_episodes} episodes")
        
        # Load a scene
        print("\n[INFO] Loading scene...")
        obs, info = env.reset(seed=1)
        
        # Get scene info
        if env.episode_data:
            house_index = env.episode_data.get("house_index", "unknown")
            target_type = env.episode_data.get("targetObjectType", "unknown")
            print(f"\n[SCENE INFO]")
            print(f"  House: {house_index}")
            print(f"  Target: {target_type}")
        
        # Move agent to create trajectory
        print("\n[INFO] Creating trajectory...")
        actions = ["moveahead", "moveahead", "rotateright", "moveahead"]
        
        for action in actions:
            action_str = f"<think>Testing</think><answer>{action}</answer>"
            obs, reward, done, info = env.step(action_str)
            print(f"  {action}: {'SUCCESS' if info.get('last_action_success') else 'FAILED'}")
        
        # Test all three map generation methods
        print("\n[INFO] Testing map generation methods...")
        
        # 1. SPOC map (should work now)
        print("  Generating SPOC map...")
        target_objects = [env.episode_data.get('targetObjectType', '')] if env.episode_data else []
        spoc_map = env.generate_spoc_map(env.agent_path, target_objects)
        
        # 2. Top-down map (should work now)
        print("  Generating top-down map...")
        top_down_map = env.get_top_down_map(include_path=True)
        
        # 3. Multi-room map (should work now)  
        print("  Generating multi-room map...")
        multi_room_map = env.get_multi_room_map(orthographic_scale=2.0)
        
        # Save results
        output_dir = "./final_map_test"
        os.makedirs(output_dir, exist_ok=True)
        
        Image.fromarray(spoc_map).save(f"{output_dir}/spoc_map_fixed.png")
        Image.fromarray(top_down_map).save(f"{output_dir}/top_down_map_fixed.png")
        Image.fromarray(multi_room_map).save(f"{output_dir}/multi_room_map_fixed.png")
        
        # Create final comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'SPOC Maps - All Methods Working\nTarget: {target_type}, House: {house_index}', fontsize=16)
        
        axes[0].imshow(spoc_map)
        axes[0].set_title('SPOC Map\n(with trajectory)')
        axes[0].axis('off')
        
        axes[1].imshow(top_down_map)
        axes[1].set_title('Top-Down Map\n(with path)')
        axes[1].axis('off')
        
        axes[2].imshow(multi_room_map)
        axes[2].set_title('Multi-Room Map\n(wide view)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/all_maps_working.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n[SUCCESS] All maps generated successfully!")
        print(f"Results saved to: {output_dir}/")
        print("  - spoc_map_fixed.png")
        print("  - top_down_map_fixed.png") 
        print("  - multi_room_map_fixed.png")
        print("  - all_maps_working.png")
        
        # Check if maps contain actual content (not gray)
        def is_gray_image(img_array):
            """Check if image is mostly gray (failed generation)."""
            if len(img_array.shape) == 3:
                # Check if all channels are similar (grayscale-like)
                std_across_channels = np.std(img_array, axis=2)
                return np.mean(std_across_channels) < 10
            return False
        
        print(f"\n[QUALITY CHECK]")
        print(f"  SPOC Map: {'✓ GOOD' if not is_gray_image(spoc_map) else '✗ GRAY'}")
        print(f"  Top-Down Map: {'✓ GOOD' if not is_gray_image(top_down_map) else '✗ GRAY'}")
        print(f"  Multi-Room Map: {'✓ GOOD' if not is_gray_image(multi_room_map) else '✗ GRAY'}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if 'env' in locals():
            print("\n[INFO] Closing environment...")
            env.close()

if __name__ == "__main__":
    main()