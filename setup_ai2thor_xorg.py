#!/usr/bin/env python3
"""
Set up AI2-THOR X server for SPOC training.
This handles the sudo requirement properly.
"""

import subprocess
import time
import os
import sys

def setup_xorg_server():
    """Set up AI2-THOR X server with proper sudo handling."""
    print("Setting up AI2-THOR X Server for SPOC Training")
    print("="*60)
    
    print("This will:")
    print("1. Start AI2-THOR's built-in X11 server (requires sudo)")
    print("2. Test that AI2-THOR rendering works")
    print("3. Test your SPOC environment")
    print("4. Provide instructions for your training script")
    
    print(f"\nNote: The sudo password is needed because AI2-THOR needs to:")
    print("- Create a virtual display")
    print("- Configure GPU access")
    print("- Set up OpenGL context")
    
    # Get user confirmation
    response = input(f"\nProceed with setup? (y/N): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        return False
    
    print(f"\nStarting AI2-THOR X server...")
    print("You will be prompted for your sudo password...")
    
    try:
        # Start X server with proper terminal interaction
        result = subprocess.run(['ai2thor-xorg', 'start'], 
                               input='', text=True, timeout=60)
        
        if result.returncode != 0:
            print("✗ Failed to start X server")
            return False
        
        print("✓ AI2-THOR X server started successfully!")
        
        # Wait a moment for server to be ready
        time.sleep(3)
        
        # Test AI2-THOR
        print("\nTesting AI2-THOR with X server...")
        import ai2thor.controller
        
        controller = ai2thor.controller.Controller(
            platform="Linux64",
            width=224,
            height=224,
            agentMode="stretch"
        )
        
        event = controller.reset(scene="FloorPlan1_physics")
        
        if event.frame is not None:
            print(f"✓ AI2-THOR works! Frame shape: {event.frame.shape}")
            success = True
        else:
            print("✗ AI2-THOR still returns None frame")
            success = False
        
        controller.stop()
        
        if success:
            # Test SPOC
            success = test_spoc_integration()
        
        return success
        
    except subprocess.TimeoutExpired:
        print("✗ Timeout waiting for X server setup")
        return False
    except KeyboardInterrupt:
        print("\n✗ Setup interrupted by user")
        return False
    except Exception as e:
        print(f"✗ Error during setup: {e}")
        return False

def test_spoc_integration():
    """Test SPOC with the X server running."""
    print(f"\nTesting SPOC integration...")
    
    try:
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
        print("✓ SpocEnv initialized successfully!")
        
        # Test reset
        obs, info = env.reset(seed=42)
        print("✓ Environment reset successful!")
        print(f"Scene: {env.episode_data['scene']}")
        print(f"Instruction: {env.episode_language_instruction[:50]}...")
        
        # Test step
        action_str = "<think>Testing movement</think><answer>moveahead</answer>"
        obs, reward, done, info = env.step(action_str)
        print("✓ Step executed successfully!")
        print(f"Reward: {reward}, Action success: {info.get('last_action_success')}")
        
        env.close()
        print("✓ SPOC integration works perfectly!")
        
        return True
        
    except Exception as e:
        print(f"✗ SPOC test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_training_scripts():
    """Create modified training scripts that use the X server."""
    print(f"\nCreating training scripts...")
    
    # Create start script
    start_script = '''#!/bin/bash
# Start AI2-THOR X server for SPOC training

echo "Starting AI2-THOR X server for SPOC training..."
echo "This requires sudo to create the virtual display."

ai2thor-xorg start

if [ $? -eq 0 ]; then
    echo "✓ AI2-THOR X server started successfully!"
    echo "You can now run your SPOC training scripts."
    echo ""
    echo "When training is complete, run:"
    echo "  ./stop_ai2thor_xorg.sh"
else
    echo "✗ Failed to start AI2-THOR X server"
    exit 1
fi
'''
    
    # Create stop script
    stop_script = '''#!/bin/bash
# Stop AI2-THOR X server

echo "Stopping AI2-THOR X server..."
ai2thor-xorg stop

if [ $? -eq 0 ]; then
    echo "✓ AI2-THOR X server stopped successfully!"
else
    echo "✗ Failed to stop AI2-THOR X server"
fi
'''
    
    # Create modified training script
    training_script = '''#!/bin/bash
# Modified SPOC training script that works with AI2-THOR X server

set -e

# Check if X server is running
if ! ai2thor-xorg status > /dev/null 2>&1; then
    echo "AI2-THOR X server is not running!"
    echo "Please start it first with: ./start_ai2thor_xorg.sh"
    exit 1
fi

echo "✓ AI2-THOR X server is running"

# Interactive input for port and CUDA devices
read -p "Enter port number (default: 5000): " PORT_INPUT
PORT=${PORT_INPUT:-5000}

read -p "Enter CUDA devices (default: 0): " CUDA_DEVICES  
CUDA_DEVICES=${CUDA_DEVICES:-0}

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Extract experiment name
EXPERIMENT_NAME="spoc-grpo-$(date +%Y%m%d-%H%M%S)"

echo "Experiment name: $EXPERIMENT_NAME"
echo "Port: $PORT"
echo "CUDA Devices: $CUDA_DEVICES"

# Create directories
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

# Create server session
tmux new-session -d -s "$SERVER_SESSION"
tmux send-keys -t "$SERVER_SESSION" "conda activate vagen" C-m
tmux send-keys -t "$SERVER_SESSION" "export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" C-m
tmux send-keys -t "$SERVER_SESSION" "export SPOC_DATA_PATH=/home/jiajunliu/spoc_data/fifteen" C-m
tmux send-keys -t "$SERVER_SESSION" "python -m vagen.server.server server.port=$PORT > server.log 2>&1" C-m

echo "✓ Server session started: $SERVER_SESSION"
echo "✓ Check server.log for any issues"
echo ""
echo "To attach to server session: tmux attach-session -t $SERVER_SESSION"
echo "To kill server session: tmux kill-session -t $SERVER_SESSION"
echo ""
echo "When done training, stop the X server with: ./stop_ai2thor_xorg.sh"
'''
    
    # Write scripts
    with open('/home/jiajunliu/VAGEN/start_ai2thor_xorg.sh', 'w') as f:
        f.write(start_script)
    os.chmod('/home/jiajunliu/VAGEN/start_ai2thor_xorg.sh', 0o755)
    
    with open('/home/jiajunliu/VAGEN/stop_ai2thor_xorg.sh', 'w') as f:
        f.write(stop_script)
    os.chmod('/home/jiajunliu/VAGEN/stop_ai2thor_xorg.sh', 0o755)
    
    with open('/home/jiajunliu/VAGEN/run_spoc_training.sh', 'w') as f:
        f.write(training_script)
    os.chmod('/home/jiajunliu/VAGEN/run_spoc_training.sh', 0o755)
    
    print("Created scripts:")
    print("  start_ai2thor_xorg.sh - Start X server")
    print("  stop_ai2thor_xorg.sh  - Stop X server")
    print("  run_spoc_training.sh  - Run SPOC training")

def show_usage_instructions():
    """Show how to use the setup."""
    print(f"\n{'='*60}")
    print("SETUP COMPLETE!")
    print(f"{'='*60}")
    
    print(f"\nYour SPOC environment is now working with AI2-THOR 5.0.0!")
    
    print(f"\nTo run training:")
    print("1. Start X server: ./start_ai2thor_xorg.sh")
    print("2. Run training:   ./run_spoc_training.sh") 
    print("3. Stop X server:  ./stop_ai2thor_xorg.sh (when done)")
    
    print(f"\nFor your multi-GPU server:")
    print("- The X server only needs to run once per machine")
    print("- Multiple training jobs can share the same X server")
    print("- Remember to stop the X server when completely done")
    
    print(f"\nBenefits of this solution:")
    print("✓ Uses modern AI2-THOR 5.0.0 (supports all SPOC scenes)")
    print("✓ Official AI2-THOR headless solution")
    print("✓ No manual Xvfb or display workarounds needed")
    print("✓ Works with your existing VAGEN code")

def cleanup_on_exit():
    """Stop X server if it's running."""
    try:
        result = subprocess.run(['ai2thor-xorg', 'status'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print(f"\nStopping AI2-THOR X server...")
            subprocess.run(['ai2thor-xorg', 'stop'], capture_output=True)
    except:
        pass

def main():
    try:
        success = setup_xorg_server()
        
        if success:
            create_training_scripts()
            show_usage_instructions()
        else:
            print(f"\nSetup failed. The X server approach didn't work.")
            print("You may need to check:")
            print("- AI2-THOR installation: pip install --upgrade ai2thor")
            print("- System dependencies")
            print("- GPU drivers")
            cleanup_on_exit()
    
    except KeyboardInterrupt:
        print(f"\nSetup interrupted.")
        cleanup_on_exit()
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        cleanup_on_exit()

if __name__ == "__main__":
    main()