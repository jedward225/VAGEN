#!/usr/bin/env python3
"""
Test SPOC server exactly like run_tmux.sh does
Focus only on environment creation
"""
import os
import sys
import subprocess
import time
import requests
import signal

# Setup paths
BASE_PATH = os.path.abspath(os.path.dirname(__file__))
SCRIPT_DIR = os.path.join(BASE_PATH, "scripts/examples/masked_grpo/spoc/grounding_worldmodeling")

print("Testing SPOC server environment creation...")
print(f"Base path: {BASE_PATH}")
print(f"Script dir: {SCRIPT_DIR}")

# Step 1: Setup environment variables exactly like tmux does
env = os.environ.copy()
env['PYTHONPATH'] = f"{BASE_PATH}:{env.get('PYTHONPATH', '')}"
env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
env['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
env['PYTHONHASHSEED'] = '0'
env['RAY_DISABLE_DOCKER_CPU_WARNING'] = '1'
env['RAY_DISABLE_RESOURCE_AUTOSCALING'] = '1'
env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
env['SPOC_DATA_PATH'] = '/home/jiajunliu/spoc_data/fifteen'  # Local path

# Step 2: Create fake vulkaninfo
print("\nCreating fake vulkaninfo...")
os.makedirs(os.path.expanduser("~/bin"), exist_ok=True)
vulkaninfo_path = os.path.expanduser("~/bin/vulkaninfo")
with open(vulkaninfo_path, 'w') as f:
    f.write('#!/bin/bash\necho "Fake vulkaninfo"\nexit 0\n')
os.chmod(vulkaninfo_path, 0o755)
env['PATH'] = f"{os.path.expanduser('~/bin')}:{env['PATH']}"

# Test vulkaninfo
try:
    result = subprocess.run(['vulkaninfo'], capture_output=True, text=True, env=env)
    print(f"Vulkaninfo test: {'SUCCESS' if result.returncode == 0 else 'FAILED'}")
except Exception as e:
    print(f"Vulkaninfo test failed: {e}")

# Step 3: Start server with xvfb-run exactly like tmux does
print("\nStarting server with xvfb-run...")
if 'DISPLAY' in env:
    del env['DISPLAY']  # unset DISPLAY

server_cmd = [
    'xvfb-run', '-a', '-s', '-screen 0 1024x768x24',
    'python', '-m', 'vagen.server.server', 'server.port=5001'
]

# Start server process
server_process = subprocess.Popen(
    server_cmd,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

print("Server starting... (PID: {})".format(server_process.pid))
print("Waiting 15 seconds for server initialization...")

# Print server output for debugging
def print_server_output():
    import select
    while True:
        if server_process.poll() is not None:
            print("Server process died!")
            return False
        
        ready, _, _ = select.select([server_process.stdout], [], [], 0.1)
        if ready:
            line = server_process.stdout.readline()
            if line:
                print(f"[SERVER] {line.strip()}")
        else:
            break
    return True

# Monitor server startup
start_time = time.time()
while time.time() - start_time < 15:
    if not print_server_output():
        break
    time.sleep(0.5)

# Step 4: Test environment creation via API
print("\n\nTesting environment creation via API...")
base_url = "http://localhost:5001"

# Test server health
try:
    response = requests.get(f"{base_url}/health", timeout=5)
    print(f"Server health check: {response.status_code}")
    if response.status_code == 200:
        health_data = response.json()
        print(f"  Registered envs: {health_data.get('registered_envs', [])}")
        print(f"  Active services: {health_data.get('active_services', [])}")
except Exception as e:
    print(f"Server not responding: {e}")
    print_server_output()
    server_process.terminate()
    sys.exit(1)

# Create all val1-val4 environments in batch
print(f"\nCreating environments val1-val4 in batch...")

ids2configs = {}
for env_id in ['val1', 'val2', 'val3', 'val4']:
    ids2configs[env_id] = {
        "env_name": "spoc",
        "env_config": {
            "data_path": "/home/jiajunliu/spoc_data/fifteen",
            "task_type": "FetchType",
            "chores_split": "train",
            "prompt_format": "grounding_worldmodeling",
            "max_actions_per_step": 3,
            "action_sep": ","
        }
    }

try:
    response = requests.post(f"{base_url}/environments", json={"ids2configs": ids2configs}, timeout=60)
    print(f"  Batch creation status: {response.status_code}")
    print(f"  Response: {response.text}")
    
    if response.status_code == 200:
        print("  ✓ Environments created successfully!")
        
        # Test reset for each environment
        for env_id in ['val1', 'val2', 'val3', 'val4']:
            print(f"\n  Testing reset for {env_id}...")
            reset_response = requests.post(
                f"{base_url}/reset/{env_id}", 
                json={"seed": 42},
                timeout=10
            )
            print(f"    Reset status: {reset_response.status_code}")
            if reset_response.status_code != 200:
                print(f"    Reset error: {reset_response.text}")
    else:
        print(f"  ✗ Environment creation failed: {response.text}")
        
except Exception as e:
    print(f"  Failed: {e}")

# Print any remaining server output
print("\n\nFinal server output:")
print_server_output()

# Cleanup
print("\nShutting down server...")
server_process.terminate()
server_process.wait()

print("\nTest complete.")