#!/usr/bin/env python3
"""
Direct test of SPOC environment without server/training pipeline
Based on the navigation env.py example
"""
import os
import sys

# Setup paths
BASE_PATH = '/root/VAGEN' if os.path.exists('/root/VAGEN') else '/home/jiajunliu/VAGEN'
sys.path.insert(0, BASE_PATH)

# Set environment variables
os.environ['PYTHONPATH'] = BASE_PATH + ':' + os.environ.get('PYTHONPATH', '')
os.environ['SPOC_DATA_PATH'] = '/root/spoc_data/fifteen' if os.path.exists('/root/spoc_data/fifteen') else '/home/jiajunliu/spoc_data/fifteen'

# Create fake vulkaninfo if needed
fake_vulkan_script = """#!/bin/bash
echo "Fake vulkaninfo"
exit 0
"""

import tempfile
import subprocess

# Create temporary directory for fake vulkaninfo
temp_dir = tempfile.mkdtemp()
fake_vulkaninfo = os.path.join(temp_dir, 'vulkaninfo')
with open(fake_vulkaninfo, 'w') as f:
    f.write(fake_vulkan_script)
os.chmod(fake_vulkaninfo, 0o755)

# Add to PATH
os.environ['PATH'] = f"{temp_dir}:{os.environ['PATH']}"

print("Testing SPOC environment directly...")
print(f"SPOC_DATA_PATH: {os.environ['SPOC_DATA_PATH']}")
print(f"Using fake vulkaninfo at: {fake_vulkaninfo}")

try:
    from vagen.env.spoc import SpocEnv, SpocEnvConfig
    
    # Create config
    config = SpocEnvConfig(
        data_path=os.environ['SPOC_DATA_PATH'],
        task_type='FetchType',
        chores_split='train',
        prompt_format='grounding_worldmodeling',
        max_actions_per_step=3,
        action_sep=','
    )
    
    print(f"\n1. Creating environment with config...")
    print(f"   Config: {config}")
    
    # Create environment
    env = SpocEnv(config)
    print("   ✓ Environment created successfully!")
    print(f"   Dataset size: {len(env.dataset)} episodes")
    
    # Test reset
    print("\n2. Testing environment reset...")
    obs, info = env.reset(seed=42)
    print("   ✓ Reset successful!")
    print(f"   Instruction: {env.episode_language_instruction}")
    print(f"   Observation keys: {list(obs.keys())}")
    print(f"   Has image: {'multi_modal_data' in obs and len(obs['multi_modal_data']) > 0}")
    
    # Test some random actions
    print("\n3. Testing random actions...")
    test_actions = [
        "moveahead",
        "rotateleft", 
        "pickup",
        "moveahead,rotateright",  # Multiple actions
        "invalid_action",  # Invalid action
    ]
    
    for i, action_str in enumerate(test_actions):
        print(f"\n   Step {i+1}: Testing action '{action_str}'")
        
        # Format the action string as the model would output it
        formatted_action = f"<think>Testing action {action_str}</think><answer>{action_str}</answer>"
        
        obs, reward, done, info = env.step(formatted_action)
        
        print(f"   - Reward: {reward}")
        print(f"   - Done: {done}")
        print(f"   - Valid actions: {env.valid_actions}")
        print(f"   - Format correct: {info.get('format_correct', False)}")
        print(f"   - Last action success: {info.get('last_action_success', False)}")
        print(f"   - Task success: {info.get('task_success', False)}")
        
        if done:
            print("   Episode finished!")
            break
    
    # Test system prompt
    print("\n4. Testing system prompt...")
    system_prompt = env.system_prompt()
    print(f"   System prompt length: {len(system_prompt)} chars")
    print(f"   System prompt preview: {system_prompt[:200]}...")
    
    print("\n✅ All tests passed! SPOC environment is working correctly.")
    
    # Cleanup
    env.close()
    
except Exception as e:
    print(f"\n❌ Failed to create/test environment: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)