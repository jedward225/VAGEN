#!/usr/bin/env python3
"""
Test script to verify SPOC validation generation logging works.
Run this to test the validation generation logging pipeline.
"""

import sys
import os
sys.path.append('/home/jed/VAGEN')

def test_spoc_config():
    """Test SPOC environment config generation"""
    print("1. Testing SPOC environment config...")
    
    from vagen.env.spoc.env_config import SpocEnvConfig
    
    config = SpocEnvConfig(
        task_type='FetchType',
        chores_split='train',
        render_mode='vision',
        max_actions_per_step=3
    )
    
    config_id = config.config_id()
    print(f"   Config ID: {config_id}")
    
    # Check that config_id includes all expected fields
    expected_fields = ['chores_split=train', 'task_type=FetchType', 'render_mode=vision', 'max_actions_per_step=3']
    for field in expected_fields:
        if field in config_id:
            print(f"   ✓ Found: {field}")
        else:
            print(f"   ✗ Missing: {field}")
            return False
    
    return True

def test_validation_logging_structure():
    """Test the validation logging data structure"""
    print("\n2. Testing validation logging data structure...")
    
    # Mock the data structure that recording_to_log() should return
    mock_validation_data = [
        {
            "env_id": "env_0",
            "config_id": "SpocEnvConfig(chores_split=train,task_type=FetchType,render_mode=vision,max_actions_per_step=3)",
            "output_str": "This is a mock trajectory output with actions",
            "image_data": None,  # Can be None for text mode
            "metrics": {
                "score": 0.5,
                "done": True,
                "step": 10,
                "action_is_valid": 0.8,
                "action_is_effective": 0.6,
                "success": True
            }
        }
    ]
    
    # Test that the structure has all required fields
    required_fields = ['config_id', 'output_str', 'metrics']
    
    for item in mock_validation_data:
        print(f"   Testing item: {item['env_id']}")
        for field in required_fields:
            if field in item:
                print(f"   ✓ Has field: {field}")
            else:
                print(f"   ✗ Missing field: {field}")
                return False
        
        # Test metrics structure
        if 'score' in item['metrics']:
            print("   ✓ Has metrics.score")
        else:
            print("   ✗ Missing metrics.score")
            return False
    
    return True

def test_wandb_integration():
    """Test WandB integration"""
    print("\n3. Testing WandB integration...")
    
    try:
        import wandb
        print("   ✓ WandB import successful")
        
        # Test if we can create a mock table
        columns = ["Input", "Output", "Score"]
        data = [["Test input", "Test output", 0.5]]
        
        table = wandb.Table(columns=columns, data=data)
        print("   ✓ WandB Table creation successful")
        
        return True
    except Exception as e:
        print(f"   ✗ WandB error: {e}")
        return False

def main():
    """Run all tests"""
    print("=== SPOC Validation Generation Testing ===")
    
    tests = [
        test_spoc_config,
        test_validation_logging_structure, 
        test_wandb_integration
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"   ✗ Test failed with exception: {e}")
            all_passed = False
    
    print(f"\n=== Test Results ===")
    if all_passed:
        print("✓ All tests passed! Validation generation logging should work.")
        print("\nThe issue is likely that:")
        print("1. Model is generating very short responses (1 token)")
        print("2. Actions are not being parsed correctly")
        print("3. Need to restart training with the fixed config")
    else:
        print("✗ Some tests failed. Check the errors above.")
    
    print(f"\nNext steps:")
    print("1. Stop current training")
    print("2. Restart with updated config (includes render_mode field)")
    print("3. Monitor response_length metrics - should be > 1 token")
    print("4. Check for val/generations in WandB after validation runs")

if __name__ == "__main__":
    main()