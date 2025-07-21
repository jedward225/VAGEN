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
    task_type = json_spec.get("task_type", "UnknownTask")
    extras = json_spec.get("extras", {}) # Use .get for safety
    
    instruction = "complete the following task" # Default
    if task_type == "FetchType":
        instruction = f"go to the {extras.get('recepLocation', 'unknown location')} and get me a {extras.get('objectName', 'unknown object')}"
    elif task_type == "RoomVisit":
        instruction = f"go to the {extras.get('roomName', 'unknown room')}"
    elif task_type == "ObjectNavType":
        instruction = f"go to the {extras.get('objectName', 'unknown object')}"

    return {
        'instruction': instruction,
        'scene': extras.get('scene_name', 'FloorPlan1'), # Default scene if not found
        'objectName': extras.get('objectName') 
    }

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
            # Show all the paths that were searched
            searched_paths = [
                os.path.join(data_path, self.task_type, split),
                os.path.join(data_path, self.task_type)
            ]
            raise FileNotFoundError(
                f"No episodes found for task '{task_type}' in any of the searched paths:\n" +
                "\n".join([f"  - {path}" for path in searched_paths]) +
                "\nPlease check your data_path and ensure the dataset is downloaded correctly."
            )
        print(f"Dataset loaded. Found {len(self.episodes)} episodes.")

    def _map_house_index_to_scene(self, house_index):
        """
        Map SPOC's house_index to AI2-THOR's supported FloorPlan scene range.
        AI2-THOR supports: FloorPlan1-30, FloorPlan201-230, FloorPlan301-330, FloorPlan401-430
        """
        # Define supported ranges
        supported_ranges = [
            (1, 30),      # FloorPlan1-30
            (201, 230),   # FloorPlan201-230
            (301, 330),   # FloorPlan301-330
            (401, 430)    # FloorPlan401-430
        ]
        
        # If the house_index is already in a supported range, use it directly
        scene_id = house_index + 1  # Convert from 0-based to 1-based
        for start, end in supported_ranges:
            if start <= scene_id <= end:
                return scene_id
        
        # If not in supported range, map to a supported range
        # Use modulo to cycle through supported ranges
        total_supported = sum(end - start + 1 for start, end in supported_ranges)
        mapped_offset = house_index % total_supported
        
        # Find which range this offset falls into
        current_offset = 0
        for start, end in supported_ranges:
            range_size = end - start + 1
            if current_offset <= mapped_offset < current_offset + range_size:
                return start + (mapped_offset - current_offset)
            current_offset += range_size
        
        # Fallback to FloorPlan1 if something goes wrong
        return 1

    def _find_episodes(self):
        """Find JSON files and load episodes from them."""
        # First try to find JSON files based on split name
        json_path = os.path.join(self.data_path, f"{self.split}.json")
        
        if os.path.exists(json_path):
            # Load from JSON file
            print(f"Loading episodes from JSON file: {json_path}")
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                tasks = data.get('tasks', [])
                for i, task in enumerate(tasks):
                    self.episodes.append({
                        "json_data": task,
                        "episode_index": i,
                        "type": "json"
                    })
                print(f"Loaded {len(tasks)} episodes from JSON")
                return
            except Exception as e:
                print(f"Error loading JSON file {json_path}: {e}")
        
        # Fallback to original HDF5 logic if JSON not found
        print(f"JSON file not found at {json_path}, trying HDF5 fallback...")
        search_paths = [
            os.path.join(self.data_path, self.task_type, self.split, "**", "hdf5_sensors.hdf5"),
            os.path.join(self.data_path, self.task_type, "**", "hdf5_sensors.hdf5"),
        ]
        
        hdf5_files = []
        for search_path in search_paths:
            found_files = glob.glob(search_path, recursive=True)
            if found_files:
                hdf5_files = found_files
                print(f"Found HDF5 files using path pattern: {search_path}")
                break
        
        if not hdf5_files:
            print(f"No HDF5 files found in any of the search paths:")
            for path in search_paths:
                print(f"  - {path}")
        
        for hdf5_path in hdf5_files:
            try:
                with h5py.File(hdf5_path, 'r') as f:
                    for episode_key in f.keys():
                        # We will check the task_type inside __getitem__ after loading
                        self.episodes.append({
                            "hdf5_path": hdf5_path,
                            "episode_key": episode_key,
                            "type": "hdf5"
                        })
            except Exception as e:
                print(f"Warning: Could not read or process {hdf5_path}. Error: {e}")

    def __len__(self) -> int:
        """Return the total number of episodes."""
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads a single episode's data from either JSON or HDF5 format.
        """
        episode_info = self.episodes[idx]
        
        if episode_info.get("type") == "json":
            # Handle JSON data format
            task_data = episode_info["json_data"]
            
            return {
                "instruction": task_data.get("instruction", "Navigate to the target object"),
                "scene": task_data.get("scene", "FloorPlan1"),
                "agentPose": task_data.get("agentPose", {
                    "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                    "rotation": 0.0,
                    "horizon": 0.0
                }),
                "targetObjectType": task_data.get("targetObjectType", None),
                "target_position": task_data.get("target_position", None),
                "targetObjectIds": task_data.get("targetObjectIds", None),
                "object_to_hide": task_data.get("object_to_hide", [])
            }
        else:
            # Handle HDF5 data format (original code)
            with h5py.File(episode_info["hdf5_path"], 'r') as f:
                episode_group = f[episode_info["episode_key"]]
                # 1. Read the house_index (it's usually a single-element array)
                try:
                    house_index = episode_group['house_index'][0]
                except KeyError:
                    print(f"FATAL: 'house_index' key not found for episode {episode_info['episode_key']}. Cannot determine scene.")
                    # Returning a dummy that will likely fail, to avoid a hard crash
                    house_index = 0 
                
                # 2. Construct the correct scene name string from the index
                # Map SPOC's large house_index to ai2thor v5.0.0's supported FloorPlan1-430 range
                mapped_scene_index = self._map_house_index_to_scene(house_index)
                scene_name = f"FloorPlan{mapped_scene_index}_physics"

                # 3. Load and process the task spec JSON to get the instruction
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
                processed_task_spec = json_templated_to_NL_spec(task_spec_data)
                
                # 4. Extract the initial agent pose
                initial_pose_data = episode_group["last_agent_location"][0]
                agent_pose = {
                    "position": {"x": initial_pose_data[0], "y": initial_pose_data[1], "z": initial_pose_data[2]},
                    "rotation": initial_pose_data[3],
                    "horizon": initial_pose_data[4],
                }
                
                # 5. Extract target object info
                target_object_type = processed_task_spec.get("objectName", None)

                # 6. Assemble the final, CORRECT data dictionary
                return {
                    "instruction": processed_task_spec['instruction'],
                    "scene": scene_name,  # <-- Use the correctly constructed scene_name
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