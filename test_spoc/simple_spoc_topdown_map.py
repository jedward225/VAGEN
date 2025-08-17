#!/usr/bin/env python3
"""
Simple SPOC Top-Down Map Generator
Only uses the get_top_down_map method to generate maps with agent trajectories.
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

def generate_spoc_topdown_map(house_seed=1, num_actions=8, output_dir="./spoc_topdown_results"):
    """
    Generate SPOC top-down map with agent trajectory.
    
    Args:
        house_seed: Seed for scene selection
        num_actions: Number of actions for trajectory
        output_dir: Output directory for results
    """
    
    SPOC_DATA_PATH = "/home/jiajunliu/spoc_data/all"
    
    print("="*60)
    print("SPOC Simple Top-Down Map Generator")
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
        print(f"\n[INFO] Initializing SPOC environment...")
        env = SpocEnv(config)
        print(f"[SUCCESS] Environment initialized with {env.number_of_episodes} episodes")
        
        # Load scene
        print(f"\n[INFO] Loading scene with seed {house_seed}...")
        obs, info = env.reset(seed=house_seed)
        
        if env.episode_data:
            house_index = env.episode_data.get("house_index", "unknown")
            target_type = env.episode_data.get("targetObjectType", "unknown")
            instruction = env.episode_language_instruction
            
            print(f"\n[SCENE INFO]")
            print(f"  House Index: {house_index}")
            print(f"  Target Object: {target_type}")
            print(f"  Instruction: {instruction}")
            
            # Get initial agent position
            agent_pos = env.env.last_event.metadata["agent"]["position"]
            print(f"  Initial Position: x={agent_pos['x']:.2f}, z={agent_pos['z']:.2f}")
        
        # Create trajectory by moving agent around
        print(f"\n[INFO] Creating trajectory with {num_actions} actions...")
        actions = ["moveahead", "moveahead", "rotateright", "moveahead", 
                  "rotateleft", "moveahead", "rotateright", "moveahead"][:num_actions]
        
        trajectory_info = []
        for i, action in enumerate(actions):
            action_str = f"<think>Exploring scene</think><answer>{action}</answer>"
            obs, reward, done, info = env.step(action_str)
            
            success = info.get('last_action_success', False)
            agent_pos = env.env.last_event.metadata["agent"]["position"]
            
            trajectory_info.append({
                "step": i+1,
                "action": action,
                "success": success,
                "position": agent_pos,
                "reward": reward
            })
            
            print(f"  Step {i+1}: {action} -> {'✓' if success else '✗'} "
                  f"| Pos: ({agent_pos['x']:.2f}, {agent_pos['z']:.2f}) | Reward: {reward:.2f}")
            
            if done:
                print("    Task completed!")
                break
        
        # Generate the top-down map with trajectory
        print(f"\n[INFO] Generating top-down map...")
        print(f"  Agent path has {len(env.agent_path)} points")
        print(f"  Path start: {env.agent_path[0] if env.agent_path else 'None'}")
        print(f"  Path end: {env.agent_path[-1] if env.agent_path else 'None'}")
        
        # Use the top-down map method
        topdown_map = env.get_top_down_map(include_path=True, path_width=0.15)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the map
        map_filename = f"{output_dir}/topdown_map_house_{house_index}.png"
        Image.fromarray(topdown_map).save(map_filename)
        
        # Also save camera views for reference
        nav_frame = env.navigation_camera
        manip_frame = env.manipulation_camera
        
        if nav_frame is not None:
            Image.fromarray(nav_frame).save(f"{output_dir}/navigation_camera.png")
        if manip_frame is not None:
            Image.fromarray(manip_frame).save(f"{output_dir}/manipulation_camera.png")
        
        # Create a comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'SPOC Top-Down Map\nHouse: {house_index}, Target: {target_type}', fontsize=16)
        
        # Top-down map
        axes[0, 0].imshow(topdown_map)
        axes[0, 0].set_title('Top-Down Map with Trajectory')
        axes[0, 0].axis('off')
        
        # Navigation camera
        if nav_frame is not None:
            axes[0, 1].imshow(nav_frame)
            axes[0, 1].set_title('Navigation Camera')
            axes[0, 1].axis('off')
        
        # Manipulation camera
        if manip_frame is not None:
            axes[1, 0].imshow(manip_frame)
            axes[1, 0].set_title('Manipulation Camera')
            axes[1, 0].axis('off')
        
        # Trajectory information
        info_text = f"Trajectory Summary:\n"
        info_text += f"• Total Steps: {len(trajectory_info)}\n"
        info_text += f"• Successful Actions: {sum(1 for t in trajectory_info if t['success'])}\n"
        info_text += f"• Total Reward: {sum(t['reward'] for t in trajectory_info):.2f}\n\n"
        info_text += "Action Details:\n"
        for t in trajectory_info[:6]:  # Show first 6 actions
            info_text += f"  {t['step']}. {t['action']}: {'✓' if t['success'] else '✗'}\n"
        if len(trajectory_info) > 6:
            info_text += f"  ... and {len(trajectory_info) - 6} more\n"
        
        axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        summary_filename = f"{output_dir}/summary_house_{house_index}.png"
        plt.savefig(summary_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save trajectory data
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        trajectory_data = {
            "house_index": house_index,
            "target_type": target_type,
            "instruction": instruction,
            "trajectory": trajectory_info,
            "agent_path": env.agent_path
        }
        
        trajectory_filename = f"{output_dir}/trajectory_house_{house_index}.json"
        with open(trajectory_filename, 'w') as f:
            json.dump(convert_for_json(trajectory_data), f, indent=2)
        
        print(f"\n[SUCCESS] Results saved:")
        print(f"  🗺️ Map: {map_filename}")
        print(f"  📊 Summary: {summary_filename}")
        print(f"  📋 Trajectory: {trajectory_filename}")
        print(f"  📷 Camera views: {output_dir}/navigation_camera.png, manipulation_camera.png")
        
        # Quality check
        map_has_content = not is_mostly_gray(topdown_map)
        trajectory_length = len(env.agent_path)
        
        print(f"\n[QUALITY CHECK]")
        print(f"  Map Content: {'✓ GOOD' if map_has_content else '✗ GRAY/EMPTY'}")
        print(f"  Trajectory Length: {trajectory_length} points")
        print(f"  Map Dimensions: {topdown_map.shape}")
        
        return {
            "map": topdown_map,
            "trajectory": trajectory_info,
            "house_index": house_index,
            "target_type": target_type,
            "output_dir": output_dir
        }
        
    except Exception as e:
        print(f"\n[ERROR] Failed to generate map: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        if 'env' in locals():
            print(f"\n[INFO] Closing environment...")
            env.close()

def is_mostly_gray(img_array):
    """Check if image is mostly gray (indicating failed generation)."""
    if len(img_array.shape) == 3:
        # Check if all channels are similar (grayscale-like)
        std_across_channels = np.std(img_array, axis=2)
        return np.mean(std_across_channels) < 15
    return False

def main():
    """Main function for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate SPOC top-down maps')
    parser.add_argument('--seed', type=int, default=1, help='Seed for scene selection')
    parser.add_argument('--actions', type=int, default=8, help='Number of actions for trajectory')
    parser.add_argument('--output', type=str, default='./spoc_topdown_results', help='Output directory')
    
    args = parser.parse_args()
    
    result = generate_spoc_topdown_map(
        house_seed=args.seed,
        num_actions=args.actions,
        output_dir=args.output
    )
    
    if result:
        print(f"\n🎉 Successfully generated top-down map for house {result['house_index']}")
    else:
        print(f"\n❌ Failed to generate map")

if __name__ == "__main__":
    main()