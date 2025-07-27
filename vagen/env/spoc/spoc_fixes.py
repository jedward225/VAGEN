"""
SPOC Environment Fixes for VAGEN
This module contains fixes for AI2-THOR version compatibility and robust object matching.
"""

import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# SPOC's official constants from stretch_initialization_utils.py
STRETCH_COMMIT_ID = "5d0ab8ab8760eb584c5ae659c2b2b951cab23246"  # Official SPOC commit
STRETCH_WRIST_BOUND_1 = -180
STRETCH_WRIST_BOUND_2 = -90

# Load SPOC's object type mappings (from object_constants.py)
AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET = {
    "AlarmClock": "alarm_clock.n.01",
    "Apple": "apple.n.01",
    "ArmChair": "armchair.n.01",
    "BaseballBat": "baseball_bat.n.01",
    "BasketBall": "basketball.n.01",
    "Bathtub": "bathtub.n.01",
    "Bed": "bed.n.01",
    "Book": "book.n.01",
    "Bottle": "bottle.n.01",
    "Bowl": "bowl.n.01",
    "Box": "box.n.01",
    "Bread": "bread.n.01",
    "ButterKnife": "butter_knife.n.01",
    "Cabinet": "cabinet.n.01",
    "Candle": "candle.n.01",
    "Cart": "handcart.n.01",
    "CD": "compact_disk.n.01",
    "CellPhone": "cellular_telephone.n.01",
    "Chair": "chair.n.01",
    "Cloth": "fabric.n.01",
    "CoffeeMachine": "coffee_maker.n.01",
    "CounterTop": "countertop.n.01",
    "CreditCard": "credit_card.n.01",
    "Cup": "cup.n.01",
    "Desk": "desk.n.01",
    "DeskLamp": "desk_lamp.n.01",
    "DishSponge": "sponge.n.01",
    "Drawer": "drawer.n.01",
    "Dresser": "chest_of_drawers.n.01",
    "Egg": "egg.n.01",
    "Faucet": "faucet.n.01",
    "FloorLamp": "floor_lamp.n.01",
    "Fork": "fork.n.01",
    "Fridge": "refrigerator.n.01",
    "GarbageCan": "ashcan.n.01",
    "Glassbottle": "bottle.n.01",
    "HandTowel": "hand_towel.n.01",
    "HousePlant": "houseplant.n.01",
    "Kettle": "kettle.n.01",
    "KeyChain": "key_ring.n.01",
    "Knife": "knife.n.01",
    "Ladle": "ladle.n.01",
    "Laptop": "laptop.n.01",
    "LaundryHamper": "clothes_hamper.n.01",
    "Lettuce": "lettuce.n.01",
    "Microwave": "microwave.n.02",
    "Mug": "mug.n.04",
    "Newspaper": "newspaper.n.01",
    "Ottoman": "ottoman.n.03",
    "Pan": "pan.n.01",
    "Pen": "pen.n.01",
    "Pencil": "pencil.n.01",
    "PepperShaker": "saltshaker.n.01",
    "Pillow": "pillow.n.01",
    "Plate": "plate.n.04",
    "Plunger": "plunger.n.03",
    "Pot": "pot.n.01",
    "Potato": "potato.n.01",
    "RemoteControl": "remote_control.n.01",
    "Safe": "safe.n.01",
    "SaltShaker": "saltshaker.n.01",
    "Shelf": "shelf.n.01",
    "SideTable": "stand.n.04",
    "SoapBar": "bar_soap.n.01",
    "SoapBottle": "soap_dispenser.n.01",
    "Sofa": "sofa.n.01",
    "Spatula": "spatula.n.01",
    "Spoon": "spoon.n.01",
    "SprayBottle": "atomizer.n.01",
    "Statue": "statue.n.01",
    "StoveBurner": "burner.n.02",
    "TableTopDecor": "decoration.n.01",
    "TeddyBear": "teddy.n.01",
    "Television": "television_receiver.n.01",
    "TennisRacket": "tennis_racket.n.01",
    "TissueBox": "tissue.n.02",
    "Toaster": "toaster.n.02",
    "Toilet": "toilet.n.01",
    "ToiletPaper": "toilet_tissue.n.01",
    "ToiletPaperHanger": "toilet_roll_holder.n.01",
    "Tomato": "tomato.n.01",
    "Vase": "vase.n.01",
    "Watch": "watch.n.01",
    "WateringCan": "watering_can.n.01",
    "WineBottle": "wine_bottle.n.01",
}

# Common object type variations in AI2-THOR
OBJECT_TYPE_VARIATIONS = {
    "mug": ["Mug", "CoffeeMug", "Cup"],
    "bottle": ["Bottle", "WineBottle", "Glassbottle", "SprayBottle", "SoapBottle"],
    "lamp": ["DeskLamp", "FloorLamp", "Lamp"],
    "book": ["Book", "Notebook"],
    "phone": ["CellPhone", "Phone", "Telephone"],
    "knife": ["Knife", "ButterKnife"],
    "shaker": ["SaltShaker", "PepperShaker"],
    "towel": ["HandTowel", "Towel", "DishTowel"],
    "plant": ["HousePlant", "Plant", "PottedPlant"],
}


def get_spoc_compatible_thor_config() -> Dict:
    """
    Returns the correct AI2-THOR configuration for SPOC compatibility.
    This ensures we use the exact same version and settings as the official SPOC.
    """
    return {
        "gridSize": 0.15,  # AGENT_MOVEMENT_CONSTANT * 0.75
        "width": 396,      # INTEL_CAMERA_WIDTH
        "height": 224,     # INTEL_CAMERA_HEIGHT  
        "visibilityDistance": 0.8673349051766235,  # MAXIMUM_DISTANCE_ARM_FROM_AGENT_CENTER
        "visibilityScheme": "Distance",
        "fieldOfView": 59,  # INTEL_VERTICAL_FOV
        "useMassThreshold": True,
        "massThreshold": 10,
        "autoSimulation": False,
        "autoSyncTransforms": True,
        "renderInstanceSegmentation": True,
        "agentMode": "stretch",
        "renderDepthImage": False,
        "cameraNearPlane": 0.01,
        "branch": None,  # IMPORTANT: do not use branch
        "commit_id": STRETCH_COMMIT_ID,  # Use SPOC's official commit
        "server_timeout": 1000,
        "snapToGrid": False,
        "fastActionEmit": True,
    }


def is_object_type_match_spoc(object_type: str, target_type: str) -> bool:
    """
    SPOC-compatible object type matching with multiple strategies.
    This implements robust matching similar to SPOC's approach.
    
    Args:
        object_type: The actual object type from AI2-THOR (e.g., "Mug_1", "CoffeeMug")
        target_type: The target object type to match (e.g., "mug", "Mug")
        
    Returns:
        True if the object types match according to SPOC rules
    """
    # Handle empty strings
    if not object_type or not target_type:
        return False
        
    # Normalize both types
    obj_type_lower = object_type.lower().strip()
    target_type_lower = target_type.lower().strip()
    
    # Remove common AI2-THOR suffixes and prefixes
    obj_base = obj_type_lower.split('|')[0].split('_')[0]
    target_base = target_type_lower.split('|')[0].split('_')[0]
    
    # Strategy 1: Exact match after normalization
    if obj_base == target_base:
        return True
    
    # Strategy 2: Check object type variations
    for category, variations in OBJECT_TYPE_VARIATIONS.items():
        variations_lower = [v.lower() for v in variations]
        if target_base in variations_lower and obj_base in variations_lower:
            return True
    
    # Strategy 3: WordNet synset matching (if available)
    obj_synset = AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET.get(object_type.split('_')[0], None)
    target_synset = AI2THOR_OBJECT_TYPE_TO_WORDNET_SYNSET.get(target_type.split('_')[0], None)
    if obj_synset and target_synset and obj_synset == target_synset:
        return True
    
    # Strategy 4: Substring matching (careful with this)
    # Only if one is clearly a substring of the other
    if len(target_base) >= 3 and len(obj_base) >= 3:  # Avoid matching single/double letters
        if target_base in obj_base or obj_base in target_base:
            return True
    
    # Strategy 5: Handle specific AI2-THOR naming conventions
    # e.g., "CoffeeMug" should match "Mug"
    if target_base == "mug" and "mug" in obj_type_lower:
        return True
    if target_base == "bottle" and "bottle" in obj_type_lower:
        return True
    if target_base == "lamp" and "lamp" in obj_type_lower:
        return True
    
    # Strategy 6: Remove common words and retry
    common_words = {"the", "a", "an", "of", "with", "coffee", "wine", "hand", "dish", "floor", "desk"}
    obj_words = set(obj_base.replace('_', ' ').split()) - common_words
    target_words = set(target_base.replace('_', ' ').split()) - common_words
    
    if obj_words and target_words and obj_words.intersection(target_words):
        return True
    
    return False


def validate_scene_name(scene_name: str) -> Tuple[bool, str]:
    """
    Validates if a scene name exists in AI2-THOR.
    Returns (is_valid, corrected_scene_name).
    """
    # Strip _physics suffix if present (AI2-THOR doesn't use this)
    clean_scene_name = scene_name.replace("_physics", "")
    
    # Extract scene number
    import re
    match = re.search(r'FloorPlan(\d+)', clean_scene_name)
    if not match:
        return False, clean_scene_name
    
    scene_num = int(match.group(1))
    
    # AI2-THOR v5.0.0 supports FloorPlan1-30, 201-230, 301-330, 401-430
    valid_ranges = [
        (1, 30),    # Kitchen scenes
        (201, 230), # Living room scenes
        (301, 330), # Bedroom scenes
        (401, 430), # Bathroom scenes
    ]
    
    is_valid = any(start <= scene_num <= end for start, end in valid_ranges)
    
    if not is_valid:
        # Map to nearest valid scene
        if scene_num <= 30:
            corrected_num = min(max(1, scene_num), 30)
        elif scene_num <= 230:
            corrected_num = min(max(201, scene_num), 230)
        elif scene_num <= 330:
            corrected_num = min(max(301, scene_num), 330)
        else:
            corrected_num = min(max(401, scene_num), 430)
        
        corrected_scene = f"FloorPlan{corrected_num}"
        return False, corrected_scene
    
    return True, clean_scene_name


def get_wrist_rotation_delta(current_wrist: float, action: str) -> float:
    """
    Calculate wrist rotation delta respecting SPOC's bounds.
    
    Args:
        current_wrist: Current wrist rotation angle
        action: "open" or "close"
        
    Returns:
        Rotation delta that respects bounds
    """
    WRIST_ROTATION = 10  # degrees per action
    
    if action == "open":
        # Counterclockwise (negative direction)
        max_rotation = abs(current_wrist - (STRETCH_WRIST_BOUND_2 + 360))
        return -1 * min(WRIST_ROTATION, max_rotation)
    else:  # "close"
        # Clockwise (positive direction)
        max_rotation = abs(STRETCH_WRIST_BOUND_1 - current_wrist)
        return min(WRIST_ROTATION, max_rotation)


# Additional constants for arm bounds
ARM_Y_MIN = 0.0  # Minimum arm height
ARM_Y_MAX = 1.2  # Maximum arm height
ARM_Z_MIN = 0.0  # Minimum arm extension
ARM_Z_MAX = 0.5  # Maximum arm extension


def clamp_arm_position(position: Dict[str, float]) -> Dict[str, float]:
    """
    Clamp arm position to valid bounds.
    """
    return {
        "x": position["x"],  # X is typically fixed at 0
        "y": max(ARM_Y_MIN, min(ARM_Y_MAX, position["y"])),
        "z": max(ARM_Z_MIN, min(ARM_Z_MAX, position["z"]))
    }