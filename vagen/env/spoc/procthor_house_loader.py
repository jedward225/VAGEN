"""
ProcTHOR House Loader for SPOC Environment
Handles loading ProcTHOR-Objaverse house JSON files from OBJAVERSE_HOUSES_DIR
"""
import os
import json
import gzip
import warnings
from typing import Dict, List, Optional, Any
import prior

# Environment variables
OBJAVERSE_HOUSES_DIR = os.environ.get("OBJAVERSE_HOUSES_DIR")
OBJAVERSE_DATA_DIR = os.environ.get("OBJAVERSE_DATA_DIR")

if OBJAVERSE_HOUSES_DIR is None:
    warnings.warn("`OBJAVERSE_HOUSES_DIR` is not set. ProcTHOR scenes will not be available.")
else:
    OBJAVERSE_HOUSES_DIR = os.path.abspath(OBJAVERSE_HOUSES_DIR)

if OBJAVERSE_DATA_DIR is None:
    warnings.warn("`OBJAVERSE_DATA_DIR` is not set. Objaverse assets may not be available.")
else:
    OBJAVERSE_DATA_DIR = os.path.abspath(OBJAVERSE_DATA_DIR)


class ProcTHORHouseLoader:
    """
    Loads ProcTHOR-Objaverse house JSON files based on house_index from SPOC dataset.
    """
    
    def __init__(self, split: str = "train", max_houses: int = None):
        """
        Initialize the ProcTHOR house loader.
        
        Args:
            split: Dataset split ("train", "val", "test")
            max_houses: Maximum number of houses to load (None for all)
        """
        self.split = split
        self.max_houses = max_houses
        self._houses = None
        self._house_index_to_house = None
        
        if OBJAVERSE_HOUSES_DIR is None:
            raise RuntimeError(
                "OBJAVERSE_HOUSES_DIR environment variable is not set. "
                "Please set it to the directory containing ProcTHOR house files."
            )
        
        # Load houses on initialization
        self._load_houses()
    
    def _load_houses(self):
        """Load ProcTHOR houses using the prior library (same as official SPOC)."""
        print(f"[ProcTHOR] Loading houses from {OBJAVERSE_HOUSES_DIR}")
        
        # Set max houses per split (following official SPOC logic)
        if self.max_houses is None:
            max_houses_per_split = {"train": 150000, "val": 15000, "test": 15000}
        else:
            max_houses_per_split = {self.split: self.max_houses}
        
        try:
            # Check if files exist in direct path or subdirectory
            split_paths = {}
            for k in ["train", "val", "test"]:
                direct_path = os.path.join(OBJAVERSE_HOUSES_DIR, f"{k}.jsonl.gz")
                subdir_path = os.path.join(OBJAVERSE_HOUSES_DIR, "houses_2023_07_28", f"{k}.jsonl.gz")
                
                if os.path.exists(direct_path):
                    split_paths[k] = direct_path
                    print(f"[ProcTHOR] Found {k} houses at: {direct_path}")
                elif os.path.exists(subdir_path):
                    split_paths[k] = subdir_path
                    print(f"[ProcTHOR] Found {k} houses at: {subdir_path}")
                # If neither exists, still add the path for prior to handle
                else:
                    split_paths[k] = direct_path
                    print(f"[ProcTHOR] Neither {direct_path} nor {subdir_path} exists")
            
            # Load houses using prior library (same as official SPOC)
            houses_dataset = prior.load_dataset(
                dataset="spoc-data",
                entity="spoc-robot", 
                revision="local-objaverse-procthor-houses",
                path_to_splits=None,
                split_to_path=split_paths,
                max_houses_per_split=max_houses_per_split,
            )
            
            self._houses = houses_dataset[self.split]
            print(f"[ProcTHOR] Loaded {len(self._houses)} houses for split '{self.split}'")
            
        except Exception as e:
            print(f"[ProcTHOR] Error loading houses with prior: {e}")
            print(f"[ProcTHOR] Falling back to direct JSON loading...")
            self._load_houses_direct()
    
    def _load_houses_direct(self):
        """Fallback: Load houses directly from JSONL.GZ files."""
        # First try direct path
        houses_file = os.path.join(OBJAVERSE_HOUSES_DIR, f"{self.split}.jsonl.gz")
        
        # If not found, try subdirectory (common structure)
        if not os.path.exists(houses_file):
            houses_file = os.path.join(OBJAVERSE_HOUSES_DIR, "houses_2023_07_28", f"{self.split}.jsonl.gz")
        
        if not os.path.exists(houses_file):
            raise FileNotFoundError(
                f"Houses file not found: {houses_file}. "
                f"Please ensure OBJAVERSE_HOUSES_DIR is set correctly and contains {self.split}.jsonl.gz"
            )
        
        houses = []
        with gzip.open(houses_file, 'rt') as f:
            for line_num, line in enumerate(f):
                if self.max_houses and len(houses) >= self.max_houses:
                    break
                    
                try:
                    house_data = json.loads(line.strip())
                    houses.append(house_data)
                except json.JSONDecodeError as e:
                    print(f"[ProcTHOR] Warning: Could not parse line {line_num} in {houses_file}: {e}")
                    continue
        
        self._houses = houses
        print(f"[ProcTHOR] Loaded {len(houses)} houses directly from {houses_file}")
        
    def _build_house_index_mapping(self):
        """Build mapping from house_index to house JSON data."""
        if self._house_index_to_house is not None:
            return
            
        self._house_index_to_house = {}
        
        for i, house in enumerate(self._houses):
            if house is None:
                continue
                
            # The house_index should be in the house metadata
            # Try different possible keys for house index
            house_index = None
            
            if "house_index" in house:
                house_index = house["house_index"]
            elif "metadata" in house and "house_index" in house["metadata"]:
                house_index = house["metadata"]["house_index"]
            elif "index" in house:
                house_index = house["index"]
            elif "id" in house:
                house_index = house["id"]
            else:
                # If no explicit house_index, try to extract from scene name or other fields
                if "scene" in house and isinstance(house["scene"], str):
                    # Try to extract number from scene name
                    import re
                    match = re.search(r'\d+', house["scene"])
                    if match:
                        house_index = int(match.group())
                        
            if house_index is not None:
                self._house_index_to_house[house_index] = house
            
            # IMPORTANT: Also map array index to house for direct access
            # This is crucial for SPOC dataset compatibility
            self._house_index_to_house[i] = house
                
        print(f"[ProcTHOR] Built mapping for {len(self._house_index_to_house)} house indices")
        
        if len([k for k in self._house_index_to_house.keys() if not isinstance(k, int) or k >= len(self._houses)]) == 0:
            print(f"[ProcTHOR] Using array indices 0-{len(self._houses)-1} as house_index mapping")
        else:
            print(f"[ProcTHOR] Found explicit house indices: {[k for k in self._house_index_to_house.keys() if not isinstance(k, int) or k >= len(self._houses)][:5]}...")
    
    def get_house_by_index(self, house_index: int) -> Optional[Dict[str, Any]]:
        """
        Get house JSON data by house_index.
        
        Args:
            house_index: House index from SPOC dataset
            
        Returns:
            House JSON data or None if not found
        """
        if self._house_index_to_house is None:
            self._build_house_index_mapping()
            
        house = self._house_index_to_house.get(house_index)
        
        if house is None:
            # For large house indices from SPOC dataset, use modulo to map to available houses
            if house_index >= len(self._houses):
                mapped_index = house_index % len(self._houses)
                print(f"[ProcTHOR] Mapping large house_index {house_index} to {mapped_index} (modulo {len(self._houses)})")
                house = self._houses[mapped_index]
            elif 0 <= house_index < len(self._houses):
                print(f"[ProcTHOR] Using house_index {house_index} as direct array index")
                house = self._houses[house_index]
            else:
                print(f"[ProcTHOR] House index {house_index} is invalid")
                # Fallback to first house
                house = self._houses[0] if self._houses else None
                
        return house
    
    def get_available_house_indices(self) -> List[int]:
        """Get list of available house indices."""
        if self._house_index_to_house is None:
            self._build_house_index_mapping()
        return list(self._house_index_to_house.keys())
    
    def get_house_count(self) -> int:
        """Get total number of loaded houses."""
        return len(self._houses) if self._houses else 0


# Global house loader instances (singleton pattern)
_house_loaders = {}

def get_house_loader(split: str = "train", max_houses: int = None) -> ProcTHORHouseLoader:
    """
    Get or create a ProcTHOR house loader for the specified split.
    
    Args:
        split: Dataset split ("train", "val", "test")  
        max_houses: Maximum number of houses to load
        
    Returns:
        ProcTHOR house loader instance
    """
    key = (split, max_houses)
    if key not in _house_loaders:
        _house_loaders[key] = ProcTHORHouseLoader(split, max_houses)
    return _house_loaders[key]


def load_procthor_house(house_index: int, split: str = "train") -> Optional[Dict[str, Any]]:
    """
    Load a specific ProcTHOR house by house_index.
    
    Args:
        house_index: House index from SPOC dataset
        split: Dataset split ("train", "val", "test")
        
    Returns:
        House JSON data or None if not found
    """
    loader = get_house_loader(split)
    return loader.get_house_by_index(house_index)