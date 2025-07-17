#!/usr/bin/env python3
"""
Test dataset creation for SPOC environment
"""
import os
import sys

# Detect base path automatically
BASE_PATH = '/root/VAGEN' if os.path.exists('/root/VAGEN') else '/home/jiajunliu/VAGEN'
sys.path.insert(0, BASE_PATH)

# Set environment variables
os.environ['PYTHONPATH'] = BASE_PATH + ':' + os.environ.get('PYTHONPATH', '')
os.environ['SPOC_DATA_PATH'] = '/root/spoc_data/fifteen' if os.path.exists('/root/spoc_data/fifteen') else '/home/jiajunliu/spoc_data/fifteen'

def test_dataset_creation():
    try:
        print("Testing dataset creation...")
        print(f"Base path: {BASE_PATH}")
        
        # Import the create_dataset function
        from vagen.env.create_dataset import create_dataset_from_yaml
        
        yaml_path = os.path.join(BASE_PATH, 'scripts/examples/masked_grpo/spoc/grounding_worldmodeling/env_config.yaml')
        train_path = '/tmp/test_train.parquet'
        test_path = '/tmp/test_test.parquet'
        
        print(f"YAML path: {yaml_path}")
        print(f"Train path: {train_path}")
        print(f"Test path: {test_path}")
        print(f"SPOC_DATA_PATH: {os.environ.get('SPOC_DATA_PATH')}")
        
        # Test dataset creation
        result = create_dataset_from_yaml(
            yaml_file_path=yaml_path,
            force_gen=True,
            seed=42,
            train_path=train_path,
            test_path=test_path
        )
        
        print(f"Dataset creation result: {result}")
        
        # Check if files were created
        if os.path.exists(train_path):
            print(f"✓ Train file created: {train_path}")
            print(f"  Size: {os.path.getsize(train_path)} bytes")
        else:
            print(f"✗ Train file not created: {train_path}")
            
        if os.path.exists(test_path):
            print(f"✓ Test file created: {test_path}")
            print(f"  Size: {os.path.getsize(test_path)} bytes")
        else:
            print(f"✗ Test file not created: {test_path}")
            
    except Exception as e:
        print(f"Error in dataset creation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_creation()