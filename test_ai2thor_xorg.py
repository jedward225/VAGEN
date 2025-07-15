#!/usr/bin/env python3
"""
Test using AI2-THOR's built-in X11 server solution.
"""

import subprocess
import os
import time

def test_ai2thor_xorg():
    """Test AI2-THOR with its built-in xorg server."""
    print("Testing AI2-THOR with built-in X11 server")
    print("="*60)
    
    # Check if ai2thor-xorg command is available
    try:
        result = subprocess.run(['which', 'ai2thor-xorg'], capture_output=True, text=True)
        if result.returncode != 0:
            print("✗ ai2thor-xorg command not found")
            print("Try: pip install --upgrade ai2thor")
            return False
        print("✓ ai2thor-xorg command found")
    except Exception as e:
        print(f"✗ Error checking ai2thor-xorg: {e}")
        return False
    
    # Start the X11 server
    print("\nStarting AI2-THOR X11 server...")
    try:
        # Start the X server in background
        xorg_process = subprocess.Popen(['ai2thor-xorg', 'start'], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE,
                                       text=True)
        
        # Wait a bit for it to start
        time.sleep(5)
        
        # Check if it's running
        result = subprocess.run(['ai2thor-xorg', 'status'], 
                               capture_output=True, text=True)
        print(f"X server status: {result.stdout.strip()}")
        
        if result.returncode != 0:
            print("✗ Failed to start X server")
            return False
        
        print("✓ AI2-THOR X server started")
        
        # Now test AI2-THOR with Linux64
        print("\nTesting AI2-THOR with Linux64 platform...")
        import ai2thor.controller
        
        controller = ai2thor.controller.Controller(
            platform="Linux64",
            width=224,
            height=224,
            agentMode="stretch"
        )
        
        event = controller.reset(scene="FloorPlan1_physics")
        
        if event.frame is not None:
            print(f"✓ SUCCESS! Frame shape: {event.frame.shape}")
            success = True
        else:
            print("✗ Frame is still None")
            success = False
        
        controller.stop()
        
        # Stop the X server
        print("\nStopping AI2-THOR X server...")
        subprocess.run(['ai2thor-xorg', 'stop'], capture_output=True)
        
        return success
        
    except Exception as e:
        print(f"✗ Error testing with X server: {e}")
        # Try to stop X server in case it's running
        try:
            subprocess.run(['ai2thor-xorg', 'stop'], capture_output=True)
        except:
            pass
        return False

def test_spoc_with_xorg():
    """Test SPOC with AI2-THOR X server."""
    print(f"\n{'='*60}")
    print("TESTING SPOC WITH AI2-THOR X SERVER")
    print(f"{'='*60}")
    
    # Start X server
    print("Starting X server...")
    xorg_process = subprocess.Popen(['ai2thor-xorg', 'start'], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
    time.sleep(5)
    
    try:
        import sys
        sys.path.insert(0, '/home/jiajunliu/VAGEN')
        os.environ['SPOC_DATA_PATH'] = '/home/jiajunliu/spoc_data/fifteen'
        
        from vagen.env.spoc import SpocEnv, SpocEnvConfig
        
        config = SpocEnvConfig(
            data_path=os.environ['SPOC_DATA_PATH'],
            task_type="FetchType",
            chores_split="train",
            resolution=224,
            prompt_format="free_think"
        )
        
        env = SpocEnv(config)
        print("✓ SpocEnv initialized with X server!")
        
        # Test reset
        obs, info = env.reset(seed=42)
        print("✓ Environment reset successful!")
        print(f"Scene: {env.episode_data['scene']}")
        print(f"Instruction: {env.episode_language_instruction[:60]}...")
        
        # Test step
        action_str = "<think>Testing movement</think><answer>moveahead</answer>"
        obs, reward, done, info = env.step(action_str)
        print("✓ Step executed successfully!")
        
        env.close()
        print("✓ SPOC works with AI2-THOR X server!")
        
        return True
        
    except Exception as e:
        print(f"✗ SPOC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Always stop X server
        print("\nStopping X server...")
        subprocess.run(['ai2thor-xorg', 'stop'], capture_output=True)

def create_training_script_with_xorg():
    """Create a modified training script that uses AI2-THOR X server."""
    script_content = '''#!/bin/bash
# Modified SPOC training script with AI2-THOR X server

set -e

# Interactive input for port and CUDA devices
read -p "Enter port number (default: 5000): " PORT_INPUT
PORT=${PORT_INPUT:-5000}

read -p "Enter CUDA devices (default: 0,1,2,3): " CUDA_DEVICES
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Extract experiment name from the path
EXPERIMENT_NAME=$(echo $SCRIPT_DIR | rev | cut -d'/' -f1-3 | rev | tr '/' '-')

echo "Experiment name: $EXPERIMENT_NAME"
echo "Port: $PORT"
echo "CUDA Devices: $CUDA_DEVICES"

# Create directories if they don't exist
mkdir -p "data/$EXPERIMENT_NAME"

# Find available session names
find_available_session() {
  local base_name=$1
  local count=0
  while tmux has-session -t "${base_name}${count}" 2>/dev/null; do
    count=$((count+1))
  done
  echo "${base_name}${count}"
}

SERVER_SESSION=$(find_available_session "spoc-server")
TRAIN_SESSION=$(find_available_session "spoc-train")

echo "Creating tmux sessions: $SERVER_SESSION and $TRAIN_SESSION"

# Create server session WITH AI2-THOR X SERVER
tmux new-session -d -s "$SERVER_SESSION"
tmux send-keys -t "$SERVER_SESSION" "conda activate vagen" C-m
tmux send-keys -t "$SERVER_SESSION" "export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" C-m
tmux send-keys -t "$SERVER_SESSION" "export SPOC_DATA_PATH=/home/jiajunliu/spoc_data/fifteen" C-m

# Start AI2-THOR X server first
tmux send-keys -t "$SERVER_SESSION" "echo 'Starting AI2-THOR X server...'" C-m
tmux send-keys -t "$SERVER_SESSION" "ai2thor-xorg start" C-m
tmux send-keys -t "$SERVER_SESSION" "sleep 10" C-m

# Start the VAGEN server
tmux send-keys -t "$SERVER_SESSION" "python -m vagen.server.server server.port=$PORT > server.log 2>&1" C-m

echo "Server started with AI2-THOR X server on port $PORT"
echo "Check server.log for any issues"

# The rest of your training script remains the same...
echo "To attach to server session: tmux attach-session -t $SERVER_SESSION"
echo "To stop everything: tmux kill-session -t $SERVER_SESSION && ai2thor-xorg stop"
'''
    
    with open('/home/jiajunliu/VAGEN/run_spoc_with_xorg.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('/home/jiajunliu/VAGEN/run_spoc_with_xorg.sh', 0o755)
    print("Created /home/jiajunliu/VAGEN/run_spoc_with_xorg.sh")

def main():
    success = test_ai2thor_xorg()
    
    if success:
        spoc_success = test_spoc_with_xorg()
        
        if spoc_success:
            print(f"\n{'='*60}")
            print("SUCCESS! AI2-THOR X server solution works!")
            print(f"{'='*60}")
            print("To use this in your training:")
            print("1. Start X server: ai2thor-xorg start")
            print("2. Run your training normally")
            print("3. Stop X server when done: ai2thor-xorg stop")
            print("\nI'll create a modified training script for you...")
            create_training_script_with_xorg()
        else:
            print("AI2-THOR X server works but SPOC integration failed")
    else:
        print("\nAI2-THOR X server test failed.")
        print("You may need to install prerequisites or use a different approach.")

if __name__ == "__main__":
    main()