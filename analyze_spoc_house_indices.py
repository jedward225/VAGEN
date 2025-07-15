#!/usr/bin/env python3
"""
Analyze SPOC dataset to understand house indices and scene names.
"""

import os
import sys
import h5py
import json
import glob
import numpy as np
from collections import Counter

# Set up paths
sys.path.insert(0, '/home/jiajunliu/VAGEN')
os.environ['SPOC_DATA_PATH'] = '/home/jiajunliu/spoc_data/fifteen'

def analyze_house_indices():
    """Analyze the house indices in the SPOC dataset."""
    data_path = os.environ['SPOC_DATA_PATH']
    task_type = "FetchType"
    split = "train"
    
    # Find HDF5 files
    search_path = os.path.join(data_path, task_type, split, "**", "hdf5_sensors.hdf5")
    hdf5_files = glob.glob(search_path, recursive=True)
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    house_indices = []
    sample_episodes = []
    errors = 0
    
    # Sample files to check
    files_to_check = hdf5_files[:20]  # Check first 20 files
    
    for i, hdf5_path in enumerate(files_to_check):
        print(f"\rChecking file {i+1}/{len(files_to_check)}...", end="", flush=True)
        try:
            with h5py.File(hdf5_path, 'r') as f:
                for j, episode_key in enumerate(f.keys()):
                    if j >= 5:  # Check first 5 episodes per file
                        break
                    
                    episode = f[episode_key]
                    
                    # Get house index
                    if 'house_index' in episode:
                        house_idx = episode['house_index'][0]
                        house_indices.append(house_idx)
                        
                        # Construct scene name as the task_loader does
                        scene_name = f"FloorPlan{house_idx + 1}"
                        
                        # Get some sample data
                        if len(sample_episodes) < 10:
                            # Try to get task spec
                            instruction = "Unknown task"
                            try:
                                task_spec_bytes = episode["templated_task_spec"][:].tobytes()
                                full_str = task_spec_bytes.decode('utf-8', errors='ignore')
                                # Find the JSON part
                                start_idx = full_str.find('{')
                                if start_idx != -1:
                                    # Simple extraction - find matching closing brace
                                    brace_count = 0
                                    for k, char in enumerate(full_str[start_idx:]):
                                        if char == '{':
                                            brace_count += 1
                                        elif char == '}':
                                            brace_count -= 1
                                        if brace_count == 0:
                                            json_str = full_str[start_idx:start_idx+k+1]
                                            task_data = json.loads(json_str)
                                            task_type = task_data.get('task_type', 'Unknown')
                                            extras = task_data.get('extras', {})
                                            if task_type == "FetchType":
                                                obj = extras.get('objectName', 'unknown')
                                                loc = extras.get('recepLocation', 'unknown')
                                                instruction = f"go to the {loc} and get me a {obj}"
                                            break
                            except:
                                pass
                            
                            sample_episodes.append({
                                'house_index': house_idx,
                                'scene_name': scene_name,
                                'instruction': instruction,
                                'hdf5_path': hdf5_path,
                                'episode_key': episode_key
                            })
                    else:
                        errors += 1
                        
        except Exception as e:
            print(f"\nError reading {hdf5_path}: {e}")
            errors += 1
    
    print(f"\n\n{'='*60}")
    print(f"HOUSE INDEX ANALYSIS")
    print(f"{'='*60}")
    print(f"Total episodes analyzed: {len(house_indices)}")
    print(f"Errors encountered: {errors}")
    
    if house_indices:
        house_indices = np.array(house_indices)
        print(f"\nHouse index statistics:")
        print(f"  Min: {house_indices.min()}")
        print(f"  Max: {house_indices.max()}")
        print(f"  Mean: {house_indices.mean():.2f}")
        print(f"  Unique values: {len(np.unique(house_indices))}")
        
        # Count occurrences
        counter = Counter(house_indices)
        print(f"\nMost common house indices:")
        for idx, count in counter.most_common(10):
            scene_name = f"FloorPlan{idx + 1}"
            print(f"  House index {idx} -> {scene_name}: {count} times")
        
        # Check which would be valid in old ai2thor
        print(f"\n{'='*60}")
        print("SCENE COMPATIBILITY CHECK")
        print(f"{'='*60}")
        
        # Old ai2thor has scenes 1-30, 201-230, 301-330, 401-430
        valid_ranges = [
            (0, 29),      # FloorPlan1-30
            (200, 229),   # FloorPlan201-230
            (300, 329),   # FloorPlan301-330
            (400, 429),   # FloorPlan401-430
        ]
        
        valid_count = 0
        invalid_indices = []
        
        for idx in house_indices:
            is_valid = any(start <= idx <= end for start, end in valid_ranges)
            if is_valid:
                valid_count += 1
            else:
                invalid_indices.append(idx)
        
        print(f"Episodes with valid scenes: {valid_count}/{len(house_indices)} ({100*valid_count/len(house_indices):.1f}%)")
        print(f"Episodes with invalid scenes: {len(invalid_indices)}/{len(house_indices)} ({100*len(invalid_indices)/len(house_indices):.1f}%)")
        
        if invalid_indices:
            invalid_counter = Counter(invalid_indices)
            print(f"\nMost common invalid house indices:")
            for idx, count in invalid_counter.most_common(10):
                scene_name = f"FloorPlan{idx + 1}"
                print(f"  House index {idx} -> {scene_name}: {count} times")
    
    print(f"\n{'='*60}")
    print("SAMPLE EPISODES")
    print(f"{'='*60}")
    for i, ep in enumerate(sample_episodes):
        print(f"\nEpisode {i+1}:")
        print(f"  House index: {ep['house_index']}")
        print(f"  Scene name: {ep['scene_name']}")
        print(f"  Instruction: {ep['instruction']}")
        print(f"  File: {os.path.basename(os.path.dirname(ep['hdf5_path']))}")

if __name__ == "__main__":
    analyze_house_indices()