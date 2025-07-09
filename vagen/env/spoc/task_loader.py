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

# Helper function adapted from official SPOC codebase
def json_templated_to_NL_spec(json_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a templated JSON spec into a natural language instruction."""
    task_type = json_spec["task_type"]
    
    if task_type == "FetchType":
        # Example: "go to the kitchen and get me a mug"
        instruction = f"go to the {json_spec['recepLocation']} and get me a {json_spec['objectName']}"
    elif task_type == "RoomVisit":
        # Example: "go to the kitchen"
        instruction = f"go to the {json_spec['roomName']}"
    elif task_type == "ObjectNavType":
        # Example: "go to the mug"
        instruction = f"go to the {json_spec['objectName']}"
    else:
        # Fallback for unknown task types
        instruction = "complete the following task"

    # Add scene and initial pose for environment setup
    json_spec['instruction'] = instruction
    json_spec['scene'] = json_spec['scene_name']
    json_spec['initial_pose'] = json_spec['initial_agent_pose']
    
    return json_spec

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
                             (e.g., /path/to/spoc_data/fifteen_type).
            task_type (str): The type of task to filter for (e.g., "FetchType").
            split (str): The dataset split to use ('train', 'val', etc.).
        """
        self.data_path = data_path
        self.task_type = task_type
        self.split = split
        self.episodes = []
        
        print(f"Loading SPOC dataset for task '{task_type}', split '{split}'...")
        self._find_episodes()
        if not self.episodes:
            raise FileNotFoundError(
                f"No episodes found for task '{task_type}' in '{os.path.join(data_path, split)}'. "
                "Please check your data_path and ensure the dataset is downloaded correctly."
            )
        print(f"Dataset loaded. Found {len(self.episodes)} episodes.")

    def _find_episodes(self):
        """Find all HDF5 files and index the episodes within them."""
        search_path = os.path.join(self.data_path, self.split, "**", "hdf5_sensors.hdf5")
        hdf5_files = glob.glob(search_path, recursive=True)
        
        for hdf5_path in hdf5_files:
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    for episode_key in f.keys():
                        # We will check the task_type inside __getitem__ after loading
                        self.episodes.append({
                            "hdf5_path": hdf5_path,
                            "episode_key": episode_key
                        })
            except Exception as e:
                print(f"Warning: Could not read or process {hdf5_path}. Error: {e}")

    def __len__(self) -> int:
        """Return the total number of episodes."""
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads a single episode's data from the HDF5 file.
        This now correctly interprets the SPOC data structure.
        """
        episode_info = self.episodes[idx]
        
        with h5py.File(episode_info["hdf5_path"], 'r') as f:
            episode_group = f[episode_info["episode_key"]]
            
            # 1. Load the templated_task_spec as a JSON object
            task_spec_bytes = episode_group["templated_task_spec"][:].tobytes()
            full_str = task_spec_bytes.decode('utf-8', errors='ignore')
            start_idx = full_str.find('{')
            open_braces = 0
            end_idx = -1
            if start_idx != -1:
                for i, char in enumerate(full_str[start_idx:]):
                    if char == '{': open_braces += 1
                    elif char == '}': open_braces -= 1
                    if open_braces == 0:
                        end_idx = start_idx + i
                        break
            
            if end_idx == -1:
                 raise ValueError(f"Corrupted JSON in episode {episode_info['episode_key']}")
            
            task_spec_json_str = full_str[start_idx : end_idx + 1]
            task_spec_data = json.loads(task_spec_json_str)

            # 2. Generate the natural language instruction and other metadata
            processed_task_spec = json_templated_to_NL_spec(task_spec_data["extras"])

            # 3. Extract the initial agent pose from the correct location
            initial_pose_data = episode_group["last_agent_location"][0]
            agent_pose = {
                "position": {"x": initial_pose_data[0], "y": initial_pose_data[1], "z": initial_pose_data[2]},
                "rotation": initial_pose_data[3],
                "horizon": initial_pose_data[4],
            }
            
            # 4. Extract target object info for success measurement
            target_object_type = processed_task_spec.get("objectName", None)

            return {
                "instruction": processed_task_spec['instruction'],
                "scene": processed_task_spec['scene'],
                "agentPose": agent_pose,
                "targetObjectType": target_object_type,
            }

def get_dataset(data_path: str, task_type: str, split: str) -> ChoresDataset:
    """Factory function to get the dataset instance."""
    return ChoresDataset(data_path, task_type, split)

if __name__ == '__main__':
    # Example for debugging the loader itself
    import os
    SPOC_DATA_PATH = os.environ.get("SPOC_DATA_PATH")
    if not SPOC_DATA_PATH:
        raise ValueError("Please set the SPOC_DATA_PATH environment variable.")
        
    dataset = get_dataset(SPOC_DATA_PATH, "FetchType", "train")
    print(f"Successfully loaded dataset with {len(dataset)} episodes.")
    
    # Test loading a single item
    if len(dataset) > 0:
        sample_item = dataset[0]
        print("\n--- Sample Item ---")
        print(f"Instruction: {sample_item['instruction']}")
        print(f"Scene: {sample_item['scene']}")
        print(f"Agent Pose: {sample_item['agentPose']}")
        print(f"Target Object: {sample_item['targetObjectType']}")
        print("-------------------\n") 