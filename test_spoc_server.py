#!/usr/bin/env python3
"""
Test SPOC server environment creation
"""
import os
import sys
import requests
import time

# Set up environment
BASE_PATH = '/root/VAGEN' if os.path.exists('/root/VAGEN') else '/home/jiajunliu/VAGEN'
sys.path.insert(0, BASE_PATH)

os.environ['PYTHONPATH'] = BASE_PATH + ':' + os.environ.get('PYTHONPATH', '')
os.environ['SPOC_DATA_PATH'] = '/root/spoc_data/fifteen' if os.path.exists('/root/spoc_data/fifteen') else '/home/jiajunliu/spoc_data/fifteen'

def test_server_directly():
    """Test creating SPOC environment directly without server"""
    print("Testing SPOC environment creation directly...")
    
    try:
        from vagen.env.spoc import SpocEnv, SpocEnvConfig
        
        config = SpocEnvConfig(
            data_path=os.environ['SPOC_DATA_PATH'],
            task_type='FetchType',
            chores_split='train',
            prompt_format='grounding_worldmodeling'
        )
        
        print(f"Creating environment with config: {config}")
        env = SpocEnv(config)
        print("✓ Environment created successfully!")
        
        print("\nTesting reset...")
        obs, info = env.reset(seed=42)
        print("✓ Reset successful!")
        print(f"Observation keys: {list(obs.keys())}")
        
        env.close()
        
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()

def test_server_api(port=5000):
    """Test server API endpoints"""
    print(f"\nTesting server API on port {port}...")
    
    base_url = f"http://localhost:{port}"
    
    try:
        # Test server health
        response = requests.get(f"{base_url}/")
        print(f"Server health check: {response.status_code}")
        
        # Test environment creation
        env_config = {
            "env_id": "test1",
            "env_name": "spoc",
            "env_config": {
                "data_path": os.environ['SPOC_DATA_PATH'],
                "task_type": "FetchType",
                "chores_split": "train",
                "prompt_format": "grounding_worldmodeling"
            }
        }
        
        print(f"\nCreating environment via API...")
        response = requests.post(f"{base_url}/create_env", json=env_config)
        print(f"Create environment response: {response.status_code}")
        print(f"Response content: {response.text}")
        
        if response.status_code == 200:
            # Test reset
            print("\nTesting reset via API...")
            response = requests.post(f"{base_url}/reset", json={"env_id": "test1", "seed": 42})
            print(f"Reset response: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Reset successful! Keys: {list(data.keys())}")
            else:
                print(f"Reset failed: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server on port {port}")
    except Exception as e:
        print(f"✗ Server API test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # First test direct environment creation
    test_server_directly()
    
    # Then test server API
    test_server_api(5000)