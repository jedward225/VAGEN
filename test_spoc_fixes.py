#!/usr/bin/env python3
"""
Unit tests for SPOC environment fixes.
Tests AI2-THOR version compatibility and robust object matching.
"""

import unittest
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from vagen.env.spoc.spoc_fixes import (
    get_spoc_compatible_thor_config,
    is_object_type_match_spoc,
    validate_scene_name,
    get_wrist_rotation_delta,
    clamp_arm_position,
    STRETCH_COMMIT_ID,
    STRETCH_WRIST_BOUND_1,
    STRETCH_WRIST_BOUND_2
)


class TestSPOCFixes(unittest.TestCase):
    """Test cases for SPOC environment fixes."""
    
    def test_ai2thor_version_config(self):
        """Test that we get the correct AI2-THOR configuration."""
        config = get_spoc_compatible_thor_config()
        
        # Check critical SPOC version
        self.assertEqual(config["commit_id"], STRETCH_COMMIT_ID)
        self.assertEqual(config["commit_id"], "5d0ab8ab8760eb584c5ae659c2b2b951cab23246")
        
        # Check other critical parameters
        self.assertEqual(config["agentMode"], "stretch")
        self.assertEqual(config["width"], 396)
        self.assertEqual(config["height"], 224)
        self.assertEqual(config["fieldOfView"], 59)
        self.assertIsNone(config["branch"])  # Must not use branch
        self.assertEqual(config["cameraNearPlane"], 0.01)
        self.assertEqual(config["server_timeout"], 1000)
        
    def test_object_matching_exact(self):
        """Test exact object type matching."""
        # Exact matches
        self.assertTrue(is_object_type_match_spoc("Mug", "Mug"))
        self.assertTrue(is_object_type_match_spoc("mug", "Mug"))
        self.assertTrue(is_object_type_match_spoc("Mug", "mug"))
        
    def test_object_matching_with_suffixes(self):
        """Test object matching with AI2-THOR suffixes."""
        # Handle numbered objects
        self.assertTrue(is_object_type_match_spoc("Mug_1", "Mug"))
        self.assertTrue(is_object_type_match_spoc("Mug_2", "mug"))
        self.assertTrue(is_object_type_match_spoc("Apple_3", "Apple"))
        
        # Handle pipe-separated IDs
        self.assertTrue(is_object_type_match_spoc("Mug|1|2|3", "Mug"))
        self.assertTrue(is_object_type_match_spoc("Apple|123", "apple"))
        
    def test_object_matching_variations(self):
        """Test object type variations."""
        # Coffee mug variations
        self.assertTrue(is_object_type_match_spoc("CoffeeMug", "Mug"))
        self.assertTrue(is_object_type_match_spoc("CoffeeMug", "mug"))
        self.assertTrue(is_object_type_match_spoc("Mug", "CoffeeMug"))
        
        # Bottle variations
        self.assertTrue(is_object_type_match_spoc("WineBottle", "Bottle"))
        self.assertTrue(is_object_type_match_spoc("SprayBottle", "bottle"))
        self.assertTrue(is_object_type_match_spoc("SoapBottle", "Bottle"))
        
        # Lamp variations
        self.assertTrue(is_object_type_match_spoc("DeskLamp", "Lamp"))
        self.assertTrue(is_object_type_match_spoc("FloorLamp", "lamp"))
        
    def test_object_matching_synsets(self):
        """Test WordNet synset-based matching."""
        # These should match via synsets
        self.assertTrue(is_object_type_match_spoc("CellPhone", "CellPhone"))
        self.assertTrue(is_object_type_match_spoc("Fridge", "Fridge"))
        
    def test_object_matching_negative_cases(self):
        """Test cases that should NOT match."""
        self.assertFalse(is_object_type_match_spoc("Mug", "Apple"))
        self.assertFalse(is_object_type_match_spoc("Chair", "Table"))
        self.assertFalse(is_object_type_match_spoc("Knife", "Fork"))
        self.assertFalse(is_object_type_match_spoc("Laptop", "Book"))
        
    def test_object_matching_edge_cases(self):
        """Test edge cases in object matching."""
        # Empty strings
        self.assertFalse(is_object_type_match_spoc("", "Mug"))
        self.assertFalse(is_object_type_match_spoc("Mug", ""))
        
        # Very short strings (should not match via substring)
        self.assertFalse(is_object_type_match_spoc("M", "Mug"))
        self.assertFalse(is_object_type_match_spoc("Mu", "Mug"))
        
        # Compound objects
        self.assertTrue(is_object_type_match_spoc("ButterKnife", "Knife"))
        self.assertTrue(is_object_type_match_spoc("HandTowel", "Towel"))
        
    def test_scene_validation(self):
        """Test scene name validation."""
        # Valid scenes
        valid, corrected = validate_scene_name("FloorPlan1_physics")
        self.assertTrue(valid)
        self.assertEqual(corrected, "FloorPlan1_physics")
        
        valid, corrected = validate_scene_name("FloorPlan201")
        self.assertTrue(valid)
        self.assertEqual(corrected, "FloorPlan201")
        
        # Invalid scenes - should be corrected
        valid, corrected = validate_scene_name("FloorPlan100_physics")
        self.assertFalse(valid)
        self.assertEqual(corrected, "FloorPlan201_physics")  # Clamped to living room range
        
        valid, corrected = validate_scene_name("FloorPlan500")
        self.assertFalse(valid)
        self.assertEqual(corrected, "FloorPlan430")  # Clamped to max bathroom
        
        valid, corrected = validate_scene_name("FloorPlan0")
        self.assertFalse(valid)
        self.assertEqual(corrected, "FloorPlan1")  # Clamped to min
        
    def test_wrist_rotation_bounds(self):
        """Test wrist rotation respects SPOC bounds."""
        # Test opening (counterclockwise)
        delta = get_wrist_rotation_delta(-135, "open")
        self.assertLessEqual(delta, 0)  # Should be negative
        self.assertGreaterEqual(delta, -10)  # Max 10 degrees
        
        # Test closing (clockwise)
        delta = get_wrist_rotation_delta(-135, "close")
        self.assertGreaterEqual(delta, 0)  # Should be positive
        self.assertLessEqual(delta, 10)  # Max 10 degrees
        
        # Test at bounds
        delta = get_wrist_rotation_delta(STRETCH_WRIST_BOUND_1, "close")
        self.assertEqual(delta, 0)  # Can't close more
        
        delta = get_wrist_rotation_delta(STRETCH_WRIST_BOUND_2 + 360, "open")
        self.assertEqual(delta, 0)  # Can't open more
        
    def test_arm_position_clamping(self):
        """Test arm position clamping."""
        # Within bounds
        pos = clamp_arm_position({"x": 0, "y": 0.8, "z": 0.3})
        self.assertEqual(pos["y"], 0.8)
        self.assertEqual(pos["z"], 0.3)
        
        # Exceeds bounds
        pos = clamp_arm_position({"x": 0, "y": 2.0, "z": 1.0})
        self.assertEqual(pos["y"], 1.2)  # Clamped to max
        self.assertEqual(pos["z"], 0.5)  # Clamped to max
        
        pos = clamp_arm_position({"x": 0, "y": -0.5, "z": -0.1})
        self.assertEqual(pos["y"], 0.0)  # Clamped to min
        self.assertEqual(pos["z"], 0.0)  # Clamped to min


class TestIntegration(unittest.TestCase):
    """Integration tests for SPOC fixes."""
    
    def test_config_completeness(self):
        """Test that config has all required fields."""
        config = get_spoc_compatible_thor_config()
        required_fields = [
            "gridSize", "width", "height", "visibilityDistance",
            "fieldOfView", "agentMode", "commit_id", "server_timeout"
        ]
        for field in required_fields:
            self.assertIn(field, config)
            
    def test_realistic_object_scenarios(self):
        """Test realistic object matching scenarios from SPOC."""
        # Scenario 1: Picking up any mug
        objects_in_scene = ["Mug_1", "CoffeeMug_2", "Cup_1", "Bowl_1"]
        target = "mug"
        matching = [obj for obj in objects_in_scene if is_object_type_match_spoc(obj, target)]
        self.assertIn("Mug_1", matching)
        self.assertIn("CoffeeMug_2", matching)
        self.assertNotIn("Bowl_1", matching)
        
        # Scenario 2: Picking up bottles
        objects_in_scene = ["WineBottle_1", "SprayBottle_2", "Vase_1"]
        target = "bottle"
        matching = [obj for obj in objects_in_scene if is_object_type_match_spoc(obj, target)]
        self.assertIn("WineBottle_1", matching)
        self.assertIn("SprayBottle_2", matching)
        self.assertNotIn("Vase_1", matching)


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)