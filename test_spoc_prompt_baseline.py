#!/usr/bin/env python3
"""
Test SPOC with proper grounding_worldmodeling format
"""
import os
import sys

BASE_PATH = '/root/VAGEN' if os.path.exists('/root/VAGEN') else '/home/jiajunliu/VAGEN'
sys.path.insert(0, BASE_PATH)

os.environ['PYTHONPATH'] = BASE_PATH + ':' + os.environ.get('PYTHONPATH', '')
os.environ['SPOC_DATA_PATH'] = '/root/spoc_data/fifteen' if os.path.exists('/root/spoc_data/fifteen') else '/home/jiajunliu/spoc_data/fifteen'

# Create fake vulkaninfo
import tempfile
temp_dir = tempfile.mkdtemp()
fake_vulkaninfo = os.path.join(temp_dir, 'vulkaninfo')
with open(fake_vulkaninfo, 'w') as f:
    f.write('#!/bin/bash\necho "Fake vulkaninfo"\nexit 0')
os.chmod(fake_vulkaninfo, 0o755)
os.environ['PATH'] = f"{temp_dir}:{os.environ['PATH']}"

try:
    from vagen.env.spoc import SpocEnv, SpocEnvConfig
    
    config = SpocEnvConfig(
        data_path=os.environ['SPOC_DATA_PATH'],
        task_type='FetchType',
        chores_split='train',
        prompt_format='grounding_worldmodeling',
        max_actions_per_step=3,
        action_sep=','
    )
    
    env = SpocEnv(config)
    print(f"✓ Environment created! Dataset: {len(env.dataset)} episodes")
    
    # Reset
    obs, info = env.reset(seed=42)
    print(f"\nTask: {env.episode_language_instruction}")
    
    # Test with proper grounding_worldmodeling format
    test_responses = [
        # Correct format with grounding and world modeling
        "Grounding: I see a kitchen with cabinets and a table. The arm is extended and empty.\nWorld Modeling: If I move forward, I will get closer to the table where objects might be placed.\nPlan: I should move forward to explore the room and look for the requested object.\nAction: moveahead",
        
        # Multiple actions
        "Grounding: I'm closer to the table now and can see some objects on it.\nWorld Modeling: Rotating left will give me a better view of the table surface.\nPlan: Rotate to scan for the target object.\nAction: rotateleft,rotateleft",
        
        # Pickup action
        "Grounding: I can see the target object on the table in front of me.\nWorld Modeling: Picking up the object will complete the fetch task.\nPlan: Pick up the object.\nAction: pickup"
    ]
    
    for i, response in enumerate(test_responses):
        print(f"\nStep {i+1}:")
        print(f"Response: {response[:100]}...")
        
        obs, reward, done, info = env.step(response)
        
        print(f"✓ Reward: {reward}")
        print(f"✓ Valid actions: {info.get('actions', [])}")
        print(f"✓ Format correct: {info.get('format_correct', False)}")
        print(f"✓ Task success: {info.get('task_success', False)}")
        
        if done:
            print("Episode completed!")
            break
    
    print("\n✅ SPOC environment deployment successful! Ready for training.")
    env.close()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)