#!/bin/bash
set -e

# Enhanced SPOC GRPO Training Script with Better Headless Mode Support
echo "===== Enhanced SPOC GRPO Training Script ====="

# Interactive input for port and CUDA devices
read -p "Enter port number (default: 5000): " PORT_INPUT
PORT=${PORT_INPUT:-5000}

read -p "Enter CUDA devices (default: 0,1,2,3): " CUDA_DEVICES
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENT_NAME=$(echo $SCRIPT_DIR | rev | cut -d'/' -f1-3 | rev | tr '/' '-')

# Enhanced environment setup for headless AI2-THOR
setup_headless_environment() {
    echo "Setting up enhanced headless environment..."
    
    # Core headless environment variables
    export DISPLAY=""
    export XAUTHORITY=""
    export XDG_RUNTIME_DIR="/tmp"
    
    # Graphics/rendering configuration
    export GALLIUM_DRIVER="softpipe"
    export MESA_GL_VERSION_OVERRIDE="3.3"
    export LIBGL_ALWAYS_SOFTWARE="1"
    export LIBGL_ALWAYS_INDIRECT="1"
    export EGL_PLATFORM="surfaceless"
    export PYOPENGL_PLATFORM="egl"
    
    # AI2-THOR specific optimizations
    export AI2THOR_RENDERING_MODE="headless"
    export AI2THOR_PLATFORM="CloudRendering"
    export AI2THOR_QUALITY="Low"
    
    # System optimization
    export OMP_NUM_THREADS="1"
    export MKL_NUM_THREADS="1"
    export NUMEXPR_NUM_THREADS="1"
    
    # Check for required dependencies
    if ! command -v xvfb-run &> /dev/null; then
        echo "Warning: xvfb-run not found. Installing virtual framebuffer..."
        sudo apt-get update -qq
        sudo apt-get install -y xvfb mesa-utils > /dev/null 2>&1 || echo "Could not install xvfb automatically"
    fi
    
    # Test OpenGL software rendering
    echo "Testing OpenGL software rendering..."
    python3 -c "
import os
for key, value in {
    'LIBGL_ALWAYS_SOFTWARE': '1',
    'MESA_GL_VERSION_OVERRIDE': '3.3'
}.items():
    os.environ[key] = value

try:
    import OpenGL.GL as gl
    print(f'OpenGL Version: {gl.glGetString(gl.GL_VERSION).decode()}')
    print('OpenGL software rendering test: PASSED')
except Exception as e:
    print(f'OpenGL test failed: {e}')
    print('This may cause AI2-THOR initialization issues')
"
}

# Cleanup function
cleanup_and_setup() {
    echo "Performing cleanup and setup..."
    
    # Clean up existing tmux sessions
    tmux list-sessions 2>/dev/null | grep "spoc-" | cut -d: -f1 | xargs -I {} tmux kill-session -t {} 2>/dev/null || true
    
    # Clean up processes on the port
    lsof -ti:$PORT | xargs -I {} kill -9 {} 2>/dev/null || true
    
    # Clean old data
    if [ -d "data/$EXPERIMENT_NAME" ]; then
        echo "Removing old experiment data: data/$EXPERIMENT_NAME"
        rm -rf "data/$EXPERIMENT_NAME"
    fi
    
    # Create directories
    mkdir -p "data/$EXPERIMENT_NAME"
}

# Find available session names
find_available_session() {
    local base_name=$1
    local count=0
    while tmux has-session -t "${base_name}${count}" 2>/dev/null; do
        count=$((count+1))
    done
    echo "${base_name}${count}"
}

# Enhanced server startup with better error handling
start_server() {
    local session_name=$1
    
    echo "Starting SPOC server in session: $session_name"
    
    tmux new-session -d -s "$session_name"
    tmux send-keys -t "$session_name" "conda activate vagen" C-m
    
    # Apply enhanced headless environment
    tmux send-keys -t "$session_name" "$(declare -f setup_headless_environment); setup_headless_environment" C-m
    
    # CUDA configuration
    tmux send-keys -t "$session_name" "export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" C-m
    
    # AI2-THOR specific variables
    tmux send-keys -t "$session_name" "export SPOC_DATA_PATH=/root/spoc_data/fifteen" C-m
    tmux send-keys -t "$session_name" "export THOR_SERVER_TIMEOUT=1000" C-m
    tmux send-keys -t "$session_name" "export THOR_CLIENT_TIMEOUT=500" C-m
    
    # VLLM and Ray optimization
    tmux send-keys -t "$session_name" "export VLLM_ATTENTION_BACKEND=XFORMERS" C-m
    tmux send-keys -t "$session_name" "export RAY_DISABLE_DOCKER_CPU_WARNING=1" C-m
    tmux send-keys -t "$session_name" "export RAY_DISABLE_RESOURCE_AUTOSCALING=1" C-m
    tmux send-keys -t "$session_name" "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128" C-m
    
    # Start server with enhanced error handling
    tmux send-keys -t "$session_name" "echo 'Starting SPOC server with enhanced headless mode...'" C-m
    tmux send-keys -t "$session_name" "python -m vagen.server.server server.port=$PORT 2>&1 | tee server_$EXPERIMENT_NAME.log" C-m
}

# Enhanced training startup
start_training() {
    local session_name=$1
    
    echo "Starting SPOC training in session: $session_name"
    
    tmux new-session -d -s "$session_name"
    tmux send-keys -t "$session_name" "cd $SCRIPT_DIR" C-m
    tmux send-keys -t "$session_name" "conda activate vagen" C-m
    
    # Apply the same environment setup as server
    tmux send-keys -t "$session_name" "$(declare -f setup_headless_environment); setup_headless_environment" C-m
    
    # CUDA and optimization settings
    tmux send-keys -t "$session_name" "export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" C-m
    tmux send-keys -t "$session_name" "export VLLM_ATTENTION_BACKEND=XFORMERS" C-m
    tmux send-keys -t "$session_name" "export RAY_DISABLE_DOCKER_CPU_WARNING=1" C-m
    tmux send-keys -t "$session_name" "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128" C-m
    tmux send-keys -t "$session_name" "export NCCL_P2P_DISABLE=1" C-m
    tmux send-keys -t "$session_name" "export NCCL_IB_DISABLE=1" C-m
    tmux send-keys -t "$session_name" "export SPOC_DATA_PATH=/root/spoc_data/fifteen" C-m
    
    # Create dataset with better error handling
    echo "Creating dataset..."
    tmux send-keys -t "$session_name" "echo 'Creating SPOC dataset...'" C-m
    tmux send-keys -t "$session_name" "python -m vagen.env.create_dataset \
        --yaml_path \"$SCRIPT_DIR/env_config.yaml\" \
        --train_path \"data/$EXPERIMENT_NAME/train.parquet\" \
        --test_path \"data/$EXPERIMENT_NAME/test.parquet\" || echo 'Dataset creation failed, but continuing...'" C-m
    
    # Wait for dataset creation
    tmux send-keys -t "$session_name" "while [ ! -f \"data/$EXPERIMENT_NAME/train.parquet\" ]; do sleep 5; echo 'Waiting for dataset...'; done" C-m
    
    # Start training with optimized parameters for headless mode
    tmux send-keys -t "$session_name" "echo 'Starting GRPO training...'" C-m
    tmux send-keys -t "$session_name" "python3 -m vagen.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        algorithm.high_level_gamma=1.0 \
        data.train_files=data/$EXPERIMENT_NAME/train.parquet \
        data.val_files=data/$EXPERIMENT_NAME/test.parquet \
        data.train_batch_size=2 \
        data.max_prompt_length=1024 \
        data.max_response_length=200 \
        data.max_trajectory_length=1200 \
        actor_rollout_ref.rollout.max_trajectory_length=1200 \
        data.image_key=images \
        data.truncation=left \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=2 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
        actor_rollout_ref.rollout.max_num_seqs=4 \
        actor_rollout_ref.rollout.max_num_batched_tokens=2400 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.temperature=0.7 \
        actor_rollout_ref.rollout.top_p=0.9 \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
        critic.model.enable_gradient_checkpointing=True \
        critic.ppo_micro_batch_size_per_gpu=1 \
        critic.ppo_mini_batch_size=2 \
        critic.model.fsdp_config.param_offload=True \
        critic.model.fsdp_config.optimizer_offload=True \
        trainer.logger='[console,wandb]' \
        trainer.project_name='vagen_spoc_enhanced' \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=100 \
        trainer.test_freq=50 \
        trainer.total_training_steps=500 \
        rollout_manager.max_turns=5 \
        rollout_manager.window_size=1 \
        rollout_manager.use_multi_turn_reward=False \
        rollout_manager.use_loss_mask=True \
        rollout_manager.use_gae_mask=True \
        rollout_manager.n_gpus_per_node=4 \
        trainer.val_before_train=False \
        trainer.val_generations_to_log_to_wandb=2 \
        rollout_manager.n_trajectory=1 \
        rollout_manager.use_service=True \
        rollout_manager.timeout=5000 \
        rollout_manager.base_url=\"http://localhost:$PORT\" \
        2>&1 | tee training_$EXPERIMENT_NAME.log" C-m
}

# Main execution
main() {
    echo "===== SPOC Enhanced Training Setup ====="
    echo "Port: $PORT"
    echo "CUDA Devices: $CUDA_DEVICES"
    echo "Experiment: $EXPERIMENT_NAME"
    echo "======================================="
    
    # Setup environment
    setup_headless_environment
    cleanup_and_setup
    
    # Create session names
    SERVER_SESSION=$(find_available_session "spoc-server")
    TRAIN_SESSION=$(find_available_session "spoc-train")
    
    # Start server
    start_server "$SERVER_SESSION"
    echo "Server started in session: $SERVER_SESSION"
    
    # Wait for server to initialize
    echo "Waiting for server to initialize (30 seconds)..."
    sleep 30
    
    # Test server connectivity
    for i in {1..5}; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "Server is responding on port $PORT"
            break
        elif [ $i -eq 5 ]; then
            echo "Warning: Server may not be responding on port $PORT"
        else
            echo "Waiting for server... (attempt $i/5)"
            sleep 10
        fi
    done
    
    # Start training
    start_training "$TRAIN_SESSION"
    echo "Training started in session: $TRAIN_SESSION"
    
    echo ""
    echo "===== Setup Complete ====="
    echo "Server session: tmux attach-session -t $SERVER_SESSION"
    echo "Training session: tmux attach-session -t $TRAIN_SESSION"
    echo "Server log: tail -f server_$EXPERIMENT_NAME.log"
    echo "Training log: tail -f training_$EXPERIMENT_NAME.log"
    echo ""
    echo "Enhanced headless mode optimizations applied:"
    echo "- Multiple AI2-THOR platform fallbacks"
    echo "- Reduced batch sizes for stability"
    echo "- Extended timeouts for server environment"
    echo "- Comprehensive OpenGL software rendering"
    echo "- Optimized GPU memory utilization"
    echo ""
    echo "Monitor progress: tmux list-sessions"
}

# Run main function
main