#!/usr/bin/env python3
"""
Test the fixed SpocEnv that handles scene compatibility issues.
"""

import os
import sys
sys.path.insert(0, '/home/jiajunliu/VAGEN')

# Set SPOC data path
os.environ['SPOC_DATA_PATH'] = '/home/jiajunliu/spoc_data/fifteen'

from vagen.env.spoc import SpocEnvConfig
from vagen.env.spoc.env_fix import SpocEnvFixed

print("Testing SpocEnvFixed with scene compatibility handling...")
print("="*60)

try:
    # Create config
    config = SpocEnvConfig(
        data_path=os.environ['SPOC_DATA_PATH'],
        task_type="FetchType",
        chores_split="train",
        resolution=224,
        prompt_format="free_think"
    )
    
    # Create environment with the fixed version
    env = SpocEnvFixed(config)
    print("✓ SpocEnvFixed initialized successfully!")
    
    # Test multiple resets to see if we can find valid scenes
    successful_resets = 0
    failed_resets = 0
    
    print("\nTesting 10 random resets...")
    for i in range(10):
        try:
            obs, info = env.reset()
            successful_resets += 1
            print(f"  Reset {i+1}: ✓ Success - Scene: {env.episode_data['scene']}, "
                  f"Instruction: {env.episode_language_instruction[:50]}...")
        except Exception as e:
            failed_resets += 1
            print(f"  Reset {i+1}: ✗ Failed - {str(e)}")
    
    print(f"\nReset success rate: {successful_resets}/10 ({100*successful_resets/10:.0f}%)")
    
    if successful_resets > 0:
        # Test a step
        print("\nTesting a step action...")
        action_str = "<think>Testing movement forward</think><answer>moveahead</answer>"
        obs, reward, done, info = env.step(action_str)
        print("✓ Step executed successfully!")
        print(f"  Reward: {reward}, Done: {done}")
        print(f"  Last action success: {info.get('last_action_success')}")
    
    env.close()
    print("\n✓ Environment closed successfully!")
    
    print("\n" + "="*60)
    if successful_resets > 0:
        print("SUCCESS! The fix works!")
        print(f"The environment can find and use valid scenes ({successful_resets}/10 successful).")
        print("\nTo use this fix in your code:")
        print("1. Replace 'from vagen.env.spoc import SpocEnv' with")
        print("   'from vagen.env.spoc.env_fix import SpocEnvFixed as SpocEnv'")
        print("2. The fix automatically skips invalid scenes and finds valid ones")
    else:
        print("ISSUE: No valid scenes found in 10 attempts.")
        print("The SPOC dataset may have too few scenes compatible with old ai2thor.")
    
except Exception as e:
    print(f"✗ Failed: {str(e)}")
    import traceback
    traceback.print_exc()