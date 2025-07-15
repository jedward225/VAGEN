#!/usr/bin/env python3
"""
Minimal test for SPOC environment with different rendering approaches.
This directly tests your SpocEnv class to see if it can initialize properly.
"""

import os
import sys
import subprocess
import time

def test_spoc_env_with_xvfb():
    """Test if SpocEnv can run with Xvfb."""
    print("\nTesting SpocEnv with Xvfb wrapper...")
    
    # Create a test script that will be run with xvfb-run
    test_script = """
import os
import sys
sys.path.insert(0, '/home/jiajunliu/VAGEN')

# Set SPOC data path
os.environ['SPOC_DATA_PATH'] = '/home/jiajunliu/spoc_data/fifteen'

from vagen.env.spoc import SpocEnv, SpocEnvConfig

try:
    config = SpocEnvConfig(
        data_path=os.environ['SPOC_DATA_PATH'],
        task_type="FetchType",
        chores_split="train",
        resolution=224,
        prompt_format="free_think"  # Changed to valid format
    )
    
    env = SpocEnv(config)
    print("✓ SpocEnv initialized successfully!")
    
    # Try to reset
    obs, info = env.reset(seed=42)
    print("✓ Environment reset successful!")
    print(f"✓ Observation keys: {obs.keys()}")
    print(f"✓ Episode instruction: {env.episode_language_instruction[:50]}...")
    
    # Try one step
    action_str = "<think>Testing movement</think><answer>moveahead</answer>"
    obs, reward, done, info = env.step(action_str)
    print("✓ Step executed successfully!")
    
    env.close()
    print("✓ Environment closed successfully!")
    
except Exception as e:
    print(f"✗ Failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    # Write test script to temporary file
    with open('/tmp/test_spoc_xvfb.py', 'w') as f:
        f.write(test_script)
    
    # Check if xvfb-run is available
    try:
        subprocess.run(['which', 'xvfb-run'], check=True, capture_output=True)
        print("✓ Xvfb is installed")
    except:
        print("✗ Xvfb not installed. Please install with: sudo apt-get install xvfb")
        return False
    
    # Run the test with xvfb-run
    try:
        # Use the vagen conda environment's python
        python_path = os.path.join(sys.prefix, 'bin', 'python')
        cmd = [
            'xvfb-run', '-a', '-s', '-screen 0 1024x768x24',
            python_path, '/tmp/test_spoc_xvfb.py'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        print("\nOutput:")
        print(result.stdout)
        if result.stderr:
            print("\nErrors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("✗ Test timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        return False
    finally:
        # Cleanup
        try:
            os.remove('/tmp/test_spoc_xvfb.py')
        except:
            pass

def test_spoc_env_direct():
    """Test if SpocEnv can run directly (will likely fail)."""
    print("\nTesting SpocEnv directly (without Xvfb)...")
    
    try:
        # Ensure we're using the right paths
        sys.path.insert(0, '/home/jiajunliu/VAGEN')
        os.environ['SPOC_DATA_PATH'] = '/home/jiajunliu/spoc_data/fifteen'
        
        # Try importing and creating environment
        from vagen.env.spoc import SpocEnv, SpocEnvConfig
        
        config = SpocEnvConfig(
            data_path=os.environ['SPOC_DATA_PATH'],
            task_type="FetchType", 
            chores_split="train",
            resolution=224,
            prompt_format="free_think"  # Changed to valid format
        )
        
        # This will likely fail without proper display
        env = SpocEnv(config)
        print("✓ SpocEnv initialized successfully! (Unexpected)")
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Failed as expected: {str(e)}")
        return False

def main():
    print("SPOC Environment Rendering Test")
    print("="*60)
    
    # Check environment
    if 'vagen' not in sys.prefix:
        print("⚠️  Please activate vagen environment: conda activate vagen")
        return
    
    # Check if SPOC data exists
    spoc_path = '/home/jiajunliu/spoc_data/fifteen'
    if not os.path.exists(spoc_path):
        print(f"⚠️  SPOC data not found at: {spoc_path}")
        return
    
    print(f"✓ SPOC data found at: {spoc_path}")
    print(f"✓ Python: {sys.executable}")
    
    # Run tests
    direct_success = test_spoc_env_direct()
    xvfb_success = test_spoc_env_with_xvfb()
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Direct execution: {'✓ PASSED' if direct_success else '✗ FAILED (expected)'}")
    print(f"Xvfb execution:   {'✓ PASSED' if xvfb_success else '✗ FAILED'}")
    
    if xvfb_success:
        print("\n✓ SUCCESS! The Xvfb solution works!")
        print("\nTo use this in your training:")
        print("1. Modify your run_tmux.sh to use xvfb-run for the server")
        print("2. The training script itself doesn't need Xvfb (it uses the server)")
        print("\nExample server command:")
        print("xvfb-run -a -s '-screen 0 1024x768x24' python -m vagen.server.server server.port=5000")
    else:
        print("\n✗ The Xvfb solution didn't work. Additional debugging needed.")

if __name__ == "__main__":
    main()