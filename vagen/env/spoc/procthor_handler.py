"""
ProcTHOR House Handling for SPOC Environment

This module handles loading ProcTHOR houses for SPOC evaluation.
Since SPOC uses procedurally generated houses, we need special handling
to either load them dynamically or fall back to existing scenes.
"""

import os
import json
import gzip
from typing import Dict, Any, Optional, List
from pathlib import Path


class ProcTHORHandler:
    """Handles ProcTHOR house loading and scene fallback for SPOC."""
    
    def __init__(self, houses_path: Optional[str] = None):
        """
        Initialize ProcTHOR handler.
        
        Args:
            houses_path: Path to ProcTHOR houses JSON/JSONL file
        """
        self.houses_path = houses_path
        self.houses_dict = {}
        self.fallback_scenes = self._get_fallback_scenes()
        
        # Try to load houses if path provided
        if houses_path and os.path.exists(houses_path):
            self._load_houses()
    
    def _get_fallback_scenes(self) -> List[str]:
        """Get list of fallback FloorPlan scenes for when ProcTHOR houses aren't available."""
        # These are the standard iTHOR scenes that should be available
        scenes = []
        
        # Kitchen scenes (1-30)
        scenes.extend([f"FloorPlan{i}_physics" for i in range(1, 31)])
        
        # Living room scenes (201-230)
        scenes.extend([f"FloorPlan{i}_physics" for i in range(201, 231)])
        
        # Bedroom scenes (301-330) 
        scenes.extend([f"FloorPlan{i}_physics" for i in range(301, 331)])
        
        # Bathroom scenes (401-430)
        scenes.extend([f"FloorPlan{i}_physics" for i in range(401, 431)])
        
        return scenes
    
    def _load_houses(self):
        """Load ProcTHOR houses from file."""
        try:
            if self.houses_path.endswith('.gz'):
                with gzip.open(self.houses_path, 'rt') as f:
                    if self.houses_path.endswith('.jsonl.gz'):
                        # Load JSONL format
                        for line in f:
                            house = json.loads(line.strip())
                            if 'id' in house:
                                self.houses_dict[house['id']] = house
                    else:
                        # Load JSON format
                        data = json.load(f)
                        if isinstance(data, list):
                            for house in data:
                                if 'id' in house:
                                    self.houses_dict[house['id']] = house
                        else:
                            self.houses_dict = data
            else:
                with open(self.houses_path, 'r') as f:
                    if self.houses_path.endswith('.jsonl'):
                        for line in f:
                            house = json.loads(line.strip())
                            if 'id' in house:
                                self.houses_dict[house['id']] = house
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            for house in data:
                                if 'id' in house:
                                    self.houses_dict[house['id']] = house
                        else:
                            self.houses_dict = data
            
            print(f"[ProcTHOR] Loaded {len(self.houses_dict)} houses from {self.houses_path}")
            
        except Exception as e:
            print(f"[ProcTHOR] Warning: Failed to load houses from {self.houses_path}: {e}")
    
    def get_house(self, house_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a ProcTHOR house by name.
        
        Args:
            house_name: Name like "proc_gen_train_003903"
            
        Returns:
            House dict if found, None otherwise
        """
        # Try direct lookup
        if house_name in self.houses_dict:
            return self.houses_dict[house_name]
        
        # Try extracting house ID and looking up by index
        if house_name.startswith("proc_gen_"):
            try:
                # Extract number from name like "proc_gen_train_003903" -> 3903
                parts = house_name.split("_")
                if len(parts) >= 4:
                    house_id = int(parts[-1])
                    
                    # Try different keys
                    for key in [house_id, str(house_id), f"house_{house_id}"]:
                        if key in self.houses_dict:
                            return self.houses_dict[key]
                            
            except (ValueError, IndexError):
                pass
        
        return None
    
    def get_fallback_scene(self, house_name: str, episode_idx: int = 0) -> str:
        """
        Get a fallback FloorPlan scene when ProcTHOR house isn't available.
        
        Args:
            house_name: Original house name
            episode_idx: Episode index for deterministic selection
            
        Returns:
            A valid FloorPlan scene name
        """
        # Use a deterministic mapping based on house name and episode
        if house_name and house_name.startswith("proc_gen_"):
            try:
                # Extract house number
                house_num = int(house_name.split("_")[-1])
                # Use modulo to map to available scenes
                scene_idx = (house_num + episode_idx) % len(self.fallback_scenes)
                return self.fallback_scenes[scene_idx]
            except:
                pass
        
        # Default fallback
        return self.fallback_scenes[episode_idx % len(self.fallback_scenes)]
    
    def should_use_create_house(self, scene_name: str) -> bool:
        """
        Check if a scene name requires CreateHouse action.
        
        Args:
            scene_name: Scene name to check
            
        Returns:
            True if CreateHouse should be used
        """
        return (scene_name.startswith("proc_gen_") and 
                scene_name in self.houses_dict)


class SceneMapper:
    """Maps SPOC dataset scenes to available AI2-THOR scenes."""
    
    @staticmethod
    def get_scene_for_reset(scene_name: str, house_index: Optional[int] = None, 
                           episode_idx: int = 0, procthor_handler: Optional[ProcTHORHandler] = None) -> Dict[str, Any]:
        """
        Get scene configuration for AI2-THOR reset.
        
        Args:
            scene_name: Original scene name from dataset
            house_index: House index from dataset
            episode_idx: Episode index
            procthor_handler: Optional ProcTHORHandler instance
            
        Returns:
            Dict with 'scene' and optionally 'house' for CreateHouse
        """
        # Check if we have a ProcTHOR house available
        if procthor_handler and procthor_handler.should_use_create_house(scene_name):
            house = procthor_handler.get_house(scene_name)
            if house:
                return {
                    'use_create_house': True,
                    'house': house,
                    'scene': 'Procedural'  # Base scene for procedural generation
                }
        
        # Otherwise use fallback scene
        if procthor_handler:
            fallback_scene = procthor_handler.get_fallback_scene(scene_name, episode_idx)
        else:
            # Simple fallback without handler
            fallback_scene = SceneMapper._simple_fallback(scene_name, house_index, episode_idx)
        
        return {
            'use_create_house': False,
            'scene': fallback_scene
        }
    
    @staticmethod
    def _simple_fallback(scene_name: str, house_index: Optional[int], episode_idx: int) -> str:
        """Simple fallback when no ProcTHORHandler available."""
        # Map to FloorPlan scenes based on house index
        if house_index is not None:
            # Determine room type based on patterns in SPOC
            if house_index % 4 == 0:
                # Kitchen scenes (1-30)
                scene_num = (house_index % 30) + 1
                return f"FloorPlan{scene_num}_physics"
            elif house_index % 4 == 1:
                # Living room scenes (201-230)
                scene_num = (house_index % 30) + 201
                return f"FloorPlan{scene_num}_physics"
            elif house_index % 4 == 2:
                # Bedroom scenes (301-330)
                scene_num = (house_index % 30) + 301
                return f"FloorPlan{scene_num}_physics"
            else:
                # Bathroom scenes (401-430)
                scene_num = (house_index % 30) + 401
                return f"FloorPlan{scene_num}_physics"
        
        # Default fallback
        scene_options = [
            "FloorPlan10_physics",  # Kitchen
            "FloorPlan201_physics", # Living room  
            "FloorPlan301_physics", # Bedroom
            "FloorPlan401_physics"  # Bathroom
        ]
        return scene_options[episode_idx % len(scene_options)]