#!/usr/bin/env python3
"""
Analyze SPOC dataset to understand the scene name mismatch issue.
"""

import os
import sys
import h5py
import json
import glob
from collections import Counter

# Set up paths
sys.path.insert(0, '/home/jiajunliu/VAGEN')
os.environ['SPOC_DATA_PATH'] = '/home/jiajunliu/spoc_data/fifteen'

def analyze_scenes():
    data_path = os.environ['SPOC_DATA_PATH']
    task_type = "FetchType"
    split = "train"
    
    # Find HDF5 files
    search_path = os.path.join(data_path, task_type, split, "**", "hdf5_sensors.hdf5")
    hdf5_files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    scene_counter = Counter()
    total_episodes = 0
    sample_episodes = []
    
    # Sample some episodes to understand the data structure
    for hdf5_path in hdf5_files[:5]:  # Just check first 5 files
        print(f"\nChecking: {hdf5_path}")
        try:
            with h5py.File(hdf5_path, 'r') as f:
                for i, episode_key in enumerate(f.keys()):
                    if i >= 10:  # Just check first 10 episodes per file
                        break
                    
                    episode = f[episode_key]
                    
                    # Get task spec
                    task_spec = json.loads(episode.attrs['templated_task_spec'])
                    scene_name = task_spec.get('extras', {}).get('scene_name', 'Unknown')
                    scene_counter[scene_name] += 1
                    total_episodes += 1
                    
                    if len(sample_episodes) < 5:
                        sample_episodes.append({
                            'scene': scene_name,
                            'task_type': task_spec.get('task_type'),
                            'extras': task_spec.get('extras', {})
                        })
                        
        except Exception as e:
            print(f"Error reading {hdf5_path}: {e}")
    
    print(f"\n{'='*60}")
    print(f"SCENE ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Total episodes analyzed: {total_episodes}")
    print(f"\nTop 20 most common scenes:")
    for scene, count in scene_counter.most_common(20):
        print(f"  {scene}: {count} times")
    
    print(f"\nSample episodes:")
    for i, ep in enumerate(sample_episodes):
        print(f"\n  Episode {i+1}:")
        print(f"    Scene: {ep['scene']}")
        print(f"    Task: {ep['task_type']}")
        print(f"    Object: {ep['extras'].get('objectName', 'N/A')}")
        print(f"    Location: {ep['extras'].get('recepLocation', 'N/A')}")
    
    # Check which scenes would work with the old ai2thor
    valid_scenes = [
        f"FloorPlan{i}_physics" for i in range(1, 31)
    ] + [
        f"FloorPlan{i}_physics" for i in range(201, 231)
    ] + [
        f"FloorPlan{i}_physics" for i in range(301, 331)
    ] + [
        f"FloorPlan{i}_physics" for i in range(401, 431)
    ]
    
    print(f"\n{'='*60}")
    print("SCENE COMPATIBILITY CHECK")
    print(f"{'='*60}")
    
    invalid_scenes = []
    valid_count = 0
    
    for scene, count in scene_counter.items():
        # Check if scene would be valid (approximation)
        if any(scene.startswith(f"FloorPlan{i}") for i in range(1, 431)):
            valid_count += count
        else:
            invalid_scenes.append((scene, count))
    
    print(f"Episodes with potentially valid scenes: {valid_count}/{total_episodes}")
    print(f"Episodes with invalid scenes: {total_episodes - valid_count}/{total_episodes}")
    
    if invalid_scenes:
        print(f"\nInvalid scenes found:")
        for scene, count in sorted(invalid_scenes)[:10]:
            print(f"  {scene}: {count} times")

if __name__ == "__main__":
    analyze_scenes()