"""
Task loader for SPOC environment with ChoresDataset.
This module handles loading episode data from the SPOC ChoresDataset format.
"""
import os
import json
import h5py
import glob
import numpy as np
from typing import Dict, Any, List

class ChoresDataset:
    """
    A class to handle loading data from the SPOC Chores dataset,
    which is stored in HDF5 format.
    """
    def __init__(self, data_path: str, task_type: str, split: str = "train"):
        """
        Initializes the dataset by finding all episodes.

        Args:
            data_path (str): The root directory of the downloaded SPOC dataset 
                             (e.g., /path/to/save/dir/fifteen_type).
            task_type (str): The specific task to load (e.g., "FetchType").
            split (str): The dataset split, "train" or "val".
        """
        self.data_path = data_path
        self.task_type = task_type
        self.split = split
        self.episode_paths = self._find_episodes()

        if not self.episode_paths:
            raise FileNotFoundError(
                f"No episodes found for task '{task_type}' in '{data_path}/{split}'. "
                f"Please check your path and that the data is downloaded."
            )

    def _find_episodes(self) -> List[Dict[str, Any]]:
        """Scans the data directory to find all HDF5 files and index episodes within them."""
        search_path = os.path.join(self.data_path, self.task_type, self.split, "*", "hdf5_sensors.hdf5")
        hdf5_files = glob.glob(search_path)
        
        episode_paths = []
        for hdf5_file in hdf5_files:
            house_id = os.path.basename(os.path.dirname(hdf5_file))
            try:
                with h5py.File(hdf5_file, 'r') as f:
                    # Each group in the HDF5 file is an episode
                    for episode_key in f.keys():
                        episode_paths.append({
                            "hdf5_path": hdf5_file,
                            "episode_key": episode_key,
                            "house_id": house_id,
                        })
            except Exception as e:
                print(f"Warning: Could not read {hdf5_file}. Error: {e}")
        
        return episode_paths

    def __len__(self) -> int:
        """Returns the total number of episodes."""
        return len(self.episode_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads a single episode's data from an HDF5 file.
        
        Args:
            idx (int): The index of the episode to load.
            
        Returns:
            A dictionary containing the initial state and task information for the episode.
        """
        if idx >= len(self):
            raise IndexError("Index out of range")
            
        episode_info = self.episode_paths[idx]
        
        with h5py.File(episode_info["hdf5_path"], 'r') as f:
            episode_group = f[episode_info["episode_key"]]
            
            # The data is stored as a 1D numpy array of uint8, representing a byte string.
            task_spec_byte_array = episode_group["templated_task_spec"][:]
            # Convert the array of integers back to a bytes object
            task_spec_bytes = task_spec_byte_array.tobytes()

            # Decode into a string, then strip all leading/trailing whitespace and null chars
            # This is a more robust way to handle padding issues in HDF5 files.
            task_spec_json = task_spec_bytes.decode('utf-8').strip().rstrip('\x00')

            # Now, load the cleaned JSON string
            task_spec = json.loads(task_spec_json)

            # --- Extract Key Information from the Task ---
            instruction = task_spec["instruction"]
            scene = task_spec["scene"]
            
            # Extract initial agent pose
            # last_agent_location stores pose at the *end* of each step, so index 0 is the initial pose
            initial_pose_data = episode_group["last_agent_location"][0]
            
            # Extract target object info
            target_object_type = task_spec["object_type"]
            # The target object ID and position are not easily available at the episode level
            # We will derive them if needed, but for now, focus on the instruction.

        agent_pose = {
            "position": {"x": initial_pose_data[0], "y": initial_pose_data[1], "z": initial_pose_data[2]},
            "rotation": initial_pose_data[3],
            "horizon": initial_pose_data[4],
        }

        # The SPOC dataset uses house IDs as scene names
        scene_name = episode_info["house_id"]

        return {
            'scene': scene_name,
            'agentPose': agent_pose,
            'targetObjectType': target_object_type,
            'instruction': instruction,
            'task_type': self.task_type,
            # Placeholder, as this info is not critical for starting the task
            'target_position': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 
            'targetObjectId': "Unknown",
        }

# Helper functions to be used by the environment
_cached_datasets = {}

def get_dataset(data_path: str, task_type: str, split: str) -> ChoresDataset:
    """A caching factory for ChoresDataset."""
    key = (data_path, task_type, split)
    if key not in _cached_datasets:
        print(f"Loading SPOC dataset for task '{task_type}', split '{split}'...")
        _cached_datasets[key] = ChoresDataset(data_path, task_type, split)
        print(f"Dataset loaded. Found {len(_cached_datasets[key])} episodes.")
    return _cached_datasets[key]


def load_chores_episode(data_path: str, task_type: str, split: str, idx: int) -> Dict[str, Any]:
    """
    Loads a single episode from the ChoresDataset using the dataset class.
    
    Args:
        data_path (str): The root directory of the SPOC dataset.
        task_type (str): The specific task to load.
        split (str): The dataset split ("train" or "val").
        idx (int): Episode index.
        
    Returns:
        A dictionary containing the episode data.
    """
    dataset = get_dataset(data_path, task_type, split)
    return dataset[idx]


def get_episode_count(data_path: str, task_type: str, split: str) -> int:
    """

    Gets the total number of episodes for a given task and split.
    
    Args:
        data_path (str): The root directory of the SPOC dataset.
        task_type (str): The specific task to load.
        split (str): The dataset split.
        
    Returns:
        The number of episodes.
    """
    dataset = get_dataset(data_path, task_type, split)
    return len(dataset) 