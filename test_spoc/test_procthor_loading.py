#!/usr/bin/env python3
"""
Test script to verify ProcTHOR house loading is working correctly.
This will test the modifications made to load ProcTHOR scenes instead of AI2-THOR FloorPlan scenes.
"""

import os
import sys
import traceback

# Add VAGEN to path
sys.path.append('/home/jiajunliu/VAGEN')

# Set environment variables
os.environ["SPOC_DATA_PATH"] = "/home/jiajunliu/spoc_data/all"
os.environ["OBJAVERSE_HOUSES_DIR"] = "/home/jiajunliu/objaverse_houses/houses_2023_07_28"
os.environ["OBJAVERSE_DATA_DIR"] = "/home/jiajunliu/objaverse_data"

def test_procthor_house_loader():
    """Test the ProcTHOR house loader."""
    print("=== Testing ProcTHOR House Loader ===")
    
    try:
        from vagen.env.spoc.procthor_house_loader import get_house_loader, load_procthor_house
        
        # Test getting a house loader
        print("1. Testing house loader initialization...")
        loader = get_house_loader(split="train", max_houses=10)  # Load only 10 houses for testing
        print(f"   ✓ Loaded {loader.get_house_count()} houses")
        
        # Test getting available house indices
        print("2. Testing available house indices...")
        indices = loader.get_available_house_indices()
        print(f"   ✓ Available house indices: {indices[:5]}... (showing first 5)")
        
        # Test loading a specific house
        if indices:
            test_house_index = indices[0]
            print(f"3. Testing loading house_index {test_house_index}...")
            house = load_procthor_house(test_house_index, "train")
            if house:
                print("   ✓ Successfully loaded house JSON")
                print(f"   ✓ House keys: {list(house.keys())[:5]}... (showing first 5)")
                if "metadata" in house:
                    print(f"   ✓ Has metadata with keys: {list(house['metadata'].keys())[:5]}...")
            else:
                print("   ✗ Failed to load house")
                return False
        else:
            print("   ✗ No house indices available")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ✗ Error testing ProcTHOR house loader: {e}")
        traceback.print_exc()
        return False

def test_spoc_dataset_loading():
    """Test the modified SPOC dataset loading."""
    print("\n=== Testing Modified SPOC Dataset Loading ===")
    
    try:
        from vagen.env.spoc.task_loader import get_dataset
        
        print("1. Testing dataset initialization...")
        dataset = get_dataset(
            data_path="/home/jiajunliu/spoc_data/all",
            task_type="FetchType", 
            split="train"
        )
        print(f"   ✓ Dataset loaded with {len(dataset)} episodes")
        
        print("2. Testing episode loading with ProcTHOR scenes...")
        if len(dataset) > 0:
            episode = dataset[0]  # Load first episode
            print(f"   ✓ Episode loaded successfully")
            print(f"   ✓ Episode keys: {list(episode.keys())}")
            
            # Check if we have ProcTHOR scene data
            scene_data = episode.get("scene")
            if isinstance(scene_data, dict):
                print("   ✓ Scene data is JSON (ProcTHOR format)")
                print(f"   ✓ Scene JSON keys: {list(scene_data.keys())[:5]}...")
            else:
                print(f"   ⚠ Scene data is string (AI2-THOR format): {scene_data}")
            
            # Check house_index
            house_index = episode.get("house_index")
            print(f"   ✓ House index: {house_index}")
            
            # Check agent pose
            agent_pose = episode.get("agentPose", {})
            print(f"   ✓ Agent pose: {agent_pose}")
            
            return True
        else:
            print("   ✗ No episodes in dataset")
            return False
            
    except Exception as e:
        print(f"   ✗ Error testing SPOC dataset loading: {e}")
        traceback.print_exc()
        return False

def test_spoc_environment():
    """Test the modified SPOC environment."""
    print("\n=== Testing Modified SPOC Environment ===")
    
    try:
        from vagen.env.spoc.env import SpocEnv
        from vagen.env.spoc.env_config import SpocEnvConfig
        
        print("1. Testing environment configuration...")
        env_config = SpocEnvConfig(
            data_path="/home/jiajunliu/spoc_data/all",
            task_type="FetchType",
            chores_split="train",
            prompt_format="grounding_worldmodeling",
            max_actions_per_step=3,
            action_sep=",",
            image_placeholder="<image>"
        )
        print("   ✓ Environment config created")
        
        print("2. Testing environment initialization...")
        env = SpocEnv(env_config)
        print("   ✓ Environment initialized")
        
        print("3. Testing environment reset with ProcTHOR scene...")
        obs, info = env.reset(seed=0)
        print("   ✓ Environment reset successful")
        print(f"   ✓ Observation keys: {list(obs.keys())}")
        print(f"   ✓ Info keys: {list(info.keys())}")
        
        # Check if we got proper images
        images = obs.get("multi_modal_data", {}).get("<image>", [])
        print(f"   ✓ Number of images: {len(images)}")
        
        # Close environment
        env.close()
        print("   ✓ Environment closed")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error testing SPOC environment: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing ProcTHOR Integration for SPOC Environment")
    print("=" * 60)
    
    results = []
    
    # Test 1: ProcTHOR house loader
    results.append(test_procthor_house_loader())
    
    # Test 2: Modified dataset loading
    results.append(test_spoc_dataset_loading()) 
    
    # Test 3: Modified environment
    results.append(test_spoc_environment())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    test_names = [
        "ProcTHOR House Loader",
        "Modified SPOC Dataset Loading", 
        "Modified SPOC Environment"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    all_passed = all(results)
    print(f"\nOverall Result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 ProcTHOR integration is working correctly!")
        print("You can now run your test_gpt4o_mini_spoc.py script with ProcTHOR scenes.")
    else:
        print("\n⚠ Some issues need to be resolved before using ProcTHOR scenes.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)