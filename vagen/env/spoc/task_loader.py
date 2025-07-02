"""
Task loader for SPOC environment with ChoresDataset.
This module handles loading episode data from the ChoresDataset.
"""

import os
import json
import random
from typing import Dict, Any


def load_chores_episode(split: str, task_type: str, idx: int) -> Dict[str, Any]:
    """
    Load a single episode from the ChoresDataset.
    
    Args:
        split: Dataset split (e.g., "fifteen", "val", "train")
        task_type: Task type ("Fetch", "ObjectNav", "RoomVisit", etc.)
        idx: Episode index
        
    Returns:
        Dict containing episode data:
        {
            'scene': str,                    # Scene identifier
            'agentPose': Dict,              # Agent initial pose
            'targetObjectType': str,         # Target object type
            'targetObjectId': str,          # Target object ID
            'instruction': str,             # Natural language instruction
            'task_type': str,               # Task type
            'target_position': Dict,        # Target object position
        }
    """
    
    # For now, create mock data that matches AI2-THOR format
    # TODO: Replace with actual ChoresDataset loading when data is available
    
    # Mock episode data for testing
    mock_episodes = [
        {
            'scene': 'FloorPlan1',
            'agentPose': {
                'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'rotation': 0.0,
                'horizon': 0.0
            },
            'targetObjectType': 'Apple',
            'targetObjectId': 'Apple|+01.20|+00.90|+00.42',
            'instruction': 'fetch an apple',
            'task_type': task_type,
            'target_position': {'x': 1.20, 'y': 0.90, 'z': 0.42}
        },
        {
            'scene': 'FloorPlan2', 
            'agentPose': {
                'position': {'x': -1.0, 'y': 0.0, 'z': 1.0},
                'rotation': 90.0,
                'horizon': 0.0
            },
            'targetObjectType': 'Mug',
            'targetObjectId': 'Mug|+02.10|+00.95|+01.30',
            'instruction': 'fetch a mug',
            'task_type': task_type,
            'target_position': {'x': 2.10, 'y': 0.95, 'z': 1.30}
        },
        {
            'scene': 'FloorPlan11',
            'agentPose': {
                'position': {'x': 1.5, 'y': 0.0, 'z': -0.5},
                'rotation': 180.0,
                'horizon': 30.0
            },
            'targetObjectType': 'Book',
            'targetObjectId': 'Book|+00.50|+01.20|+02.10',
            'instruction': 'fetch a book',
            'task_type': task_type,
            'target_position': {'x': 0.50, 'y': 1.20, 'z': 2.10}
        }
    ]
    
    # Return episode based on index (cycle through mock data)
    episode_idx = idx % len(mock_episodes)
    episode = mock_episodes[episode_idx].copy()
    
    # Add some randomization for variety
    random.seed(idx)
    if task_type == "Fetch":
        objects = ['Apple', 'Mug', 'Book', 'Cup', 'Plate', 'Fork']
        obj_type = random.choice(objects)
        episode['targetObjectType'] = obj_type
        episode['instruction'] = f'fetch a {obj_type.lower()}'
        # Randomize target position slightly
        x_offset = random.uniform(-0.5, 0.5)
        z_offset = random.uniform(-0.5, 0.5)
        episode['target_position']['x'] += x_offset
        episode['target_position']['z'] += z_offset
        episode['targetObjectId'] = f"{obj_type}|+{episode['target_position']['x']:.2f}|+{episode['target_position']['y']:.2f}|+{episode['target_position']['z']:.2f}"
    
    return episode


def load_real_chores_episode(data_dir: str, split: str, task_type: str, idx: int) -> Dict[str, Any]:
    """
    Load a real episode from ChoresDataset (when data becomes available).
    
    This function will be implemented when we have access to the actual 
    ChoresDataset files (hdf5_sensors.hdf5, etc.)
    
    Args:
        data_dir: Path to ChoresDataset directory
        split: Dataset split
        task_type: Task type
        idx: Episode index
        
    Returns:
        Episode data dict
    """
    # Import heavy dependencies only when needed
    try:
        import h5py
        import numpy as np
    except ImportError as e:
        raise ImportError(f"Missing dependency for real ChoresDataset: {e}")
    
    # TODO: Implement real data loading
    # 1. Read house_id_to_sub_house_id_{split}.json
    # 2. Load hdf5_sensors.hdf5 for the episode
    # 3. Extract initial_agent_location, templated_task_spec, etc.
    # 4. Convert to VAGEN format
    
    raise NotImplementedError("Real ChoresDataset loading not yet implemented")


def get_episode_count(split: str) -> int:
    """
    Get the total number of episodes in a split.
    
    Args:
        split: Dataset split
        
    Returns:
        Number of episodes
    """
    # For mock data, return a fixed count
    return 100  # TODO: Replace with actual count from dataset 