#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Interactive input for port and CUDA devices
read -p "Enter port number (default: 5000): " PORT_INPUT
PORT=${PORT_INPUT:-5000}

read -p "Enter CUDA devices (default: 0,1,2,3): " CUDA_DEVICES
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Extract experiment name from the path
# This will take the last 3 parts of the path: grounding_worldmodeling/spoc/masked_grpo
EXPERIMENT_NAME=$(echo $SCRIPT_DIR | rev | cut -d'/' -f1-3 | rev | tr '/' '-')

# --------- 磁盘清理：如已存在同名实验目录，先删除 ---------
if [ -d "data/$EXPERIMENT_NAME" ]; then
  echo "[Cleanup] Remove old data/$EXPERIMENT_NAME to free disk space"
  rm -rf "data/$EXPERIMENT_NAME"
fi
echo "Experiment name: $EXPERIMENT_NAME"

# Find available session names
find_available_session() {
  local base_name=$1
  local count=0
  
  while tmux has-session -t "${base_name}${count}" 2>/dev/null; do
    count=$((count+1))
  done
  
  echo "${base_name}${count}"
}

# Create session names
SERVER_SESSION=$(find_available_session "spoc-server")
TRAIN_SESSION=$(find_available_session "spoc-train")

echo "Creating tmux sessions: $SERVER_SESSION and $TRAIN_SESSION"
echo "Using port: $PORT"
echo "Using CUDA devices: $CUDA_DEVICES"

# Create directories if they don't exist
mkdir -p "data/$EXPERIMENT_NAME"
mkdir -p "$SCRIPT_DIR/logs"

# Create server session
tmux new-session -d -s "$SERVER_SESSION"
# Configure server session with conda and environment variables
tmux send-keys -t "$SERVER_SESSION" "conda activate vagen" C-m
# Add VAGEN root to PYTHONPATH
tmux send-keys -t "$SERVER_SESSION" "export PYTHONPATH=/root/VAGEN:\$PYTHONPATH" C-m
# headlessly，防止 AI2-THOR 在 X11 下弹窗 这个可能是val创建环境的错误来源 ↓
# tmux send-keys -t "$SERVER_SESSION" "unset DISPLAY" C-m
tmux send-keys -t "$SERVER_SESSION" "export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" C-m
tmux send-keys -t "$SERVER_SESSION" "export VLLM_ATTENTION_BACKEND=XFORMERS" C-m
tmux send-keys -t "$SERVER_SESSION" "export PYTHONHASHSEED=0" C-m
tmux send-keys -t "$SERVER_SESSION" "export RAY_DISABLE_DOCKER_CPU_WARNING=1" C-m
tmux send-keys -t "$SERVER_SESSION" "export RAY_DISABLE_RESOURCE_AUTOSCALING=1" C-m
tmux send-keys -t "$SERVER_SESSION" "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128" C-m
tmux send-keys -t "$SERVER_SESSION" "export SPOC_DATA_PATH=/root/spoc_data/fifteen" C-m
# Start the server
# headlessly, prevent AI2-THOR from using a display, which is critical for CloudRendering
tmux send-keys -t "$SERVER_SESSION" "unset DISPLAY" C-m
tmux send-keys -t "$SERVER_SESSION" "xvfb-run -a -s '-screen 0 1024x768x24' python -m vagen.server.server server.port=$PORT > $SCRIPT_DIR/logs/server.log 2>&1" C-m

# Wait for server to start
echo "Waiting for server to start on port $PORT..."
sleep 15  # Give more time for SPOC environment initialization

# Create training session
tmux new-session -d -s "$TRAIN_SESSION"
# Configure training session with conda and environment variables
tmux send-keys -t "$TRAIN_SESSION" "cd $SCRIPT_DIR" C-m
tmux send-keys -t "$TRAIN_SESSION" "conda activate vagen" C-m
# Add VAGEN root to PYTHONPATH
tmux send-keys -t "$TRAIN_SESSION" "export PYTHONPATH=/root/VAGEN:\$PYTHONPATH" C-m
tmux send-keys -t "$TRAIN_SESSION" "export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES" C-m
tmux send-keys -t "$TRAIN_SESSION" "export VLLM_ATTENTION_BACKEND=XFORMERS" C-m
tmux send-keys -t "$TRAIN_SESSION" "export PYTHONHASHSEED=0" C-m
tmux send-keys -t "$TRAIN_SESSION" "export RAY_DISABLE_DOCKER_CPU_WARNING=1" C-m
tmux send-keys -t "$TRAIN_SESSION" "export RAY_DISABLE_RESOURCE_AUTOSCALING=1" C-m
tmux send-keys -t "$TRAIN_SESSION" "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128" C-m
tmux send-keys -t "$TRAIN_SESSION" "export NCCL_P2P_DISABLE=1" C-m
tmux send-keys -t "$TRAIN_SESSION" "export NCCL_IB_DISABLE=1" C-m
tmux send-keys -t "$TRAIN_SESSION" "export MASTER_PORT=29500" C-m
tmux send-keys -t "$TRAIN_SESSION" "export SPOC_DATA_PATH=/root/spoc_data/fifteen" C-m
tmux send-keys -t "$TRAIN_SESSION" "set -x" C-m

# First create the dataset
echo "Creating dataset..."
tmux send-keys -t "$TRAIN_SESSION" "python -m vagen.env.create_dataset \\
    --yaml_path \"$SCRIPT_DIR/env_config.yaml\" \\
    --train_path \"data/$EXPERIMENT_NAME/train.parquet\" \\
    --test_path \"data/$EXPERIMENT_NAME/test.parquet\"" C-m

# Wait for dataset creation to complete
echo "Waiting for dataset creation to complete..."
tmux send-keys -t "$TRAIN_SESSION" "while [ ! -f \"data/$EXPERIMENT_NAME/train.parquet\" ]; do sleep 5; echo 'Waiting for dataset creation...'; done" C-m
tmux send-keys -t "$TRAIN_SESSION" "echo 'Dataset creation completed!'" C-m
sleep 10

# Then start the training - adapted for SPOC environment with 4 GPUs
tmux send-keys -t "$TRAIN_SESSION" "python3 -m vagen.trainer.main_ppo \\
    algorithm.adv_estimator=grpo \\
    algorithm.high_level_gamma=1.0 \\
    data.train_files=data/$EXPERIMENT_NAME/train.parquet \\
    data.val_files=data/$EXPERIMENT_NAME/test.parquet \\
    data.train_batch_size=4 \\
    data.max_prompt_length=1024 \\
    data.max_response_length=200 \\
    data.max_trajectory_length=4096 \\
    actor_rollout_ref.rollout.max_trajectory_length=4096 \\
    data.image_key=images \\
    data.truncation=left \\
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.model.use_remove_padding=True \\
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \\
    actor_rollout_ref.actor.use_kl_loss=False \\
    actor_rollout_ref.actor.kl_loss_coef=0.001 \\
    actor_rollout_ref.actor.kl_loss_type=mse \\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \\
    actor_rollout_ref.actor.fsdp_config.param_offload=True \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \\
    actor_rollout_ref.rollout.max_num_seqs=4 \\
    actor_rollout_ref.rollout.max_model_len=90000 \\
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \\
    actor_rollout_ref.rollout.enable_chunked_prefill=False \\
    actor_rollout_ref.rollout.enforce_eager=True \\
    actor_rollout_ref.rollout.free_cache_engine=True \\
    actor_rollout_ref.rollout.n=1 \\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    actor_rollout_ref.rollout.top_p=0.9 \\
    actor_rollout_ref.rollout.temperature=0.7 \\
    critic.optim.lr=1e-5 \\
    critic.model.use_remove_padding=True \\
    data.val_batch_size=4 \\
    critic.model.path=Qwen/Qwen2.5-VL-3B-Instruct \\
    critic.model.enable_gradient_checkpointing=True \\
    critic.ppo_micro_batch_size_per_gpu=1 \\
    critic.ppo_mini_batch_size=4 \\
    critic.model.fsdp_config.param_offload=True \\
    critic.model.fsdp_config.optimizer_offload=True \\
    algorithm.kl_ctrl.kl_coef=0.001 \\
    trainer.critic_warmup=0 \\
    trainer.logger='[console,wandb]' \\
    trainer.project_name='vagen_spoc' \\
    trainer.experiment_name=$EXPERIMENT_NAME \\
    trainer.n_gpus_per_node=4 \\
    trainer.nnodes=1 \\
    trainer.save_freq=200 \\
    trainer.test_freq=20 \\
    trainer.total_training_steps=1000 \\
    rollout_manager.max_turns=5 \\
    rollout_manager.window_size=1 \\
    rollout_manager.use_multi_turn_reward=False \\
    rollout_manager.use_loss_mask=True \\
    rollout_manager.use_gae_mask=True \\
    rollout_manager.n_gpus_per_node=4 \\
    trainer.val_before_train=True \\
    trainer.val_generations_to_log_to_wandb=16 \\
    rollout_manager.n_trajectory=1 \\
    rollout_manager.use_service=True \\
    rollout_manager.timeout=3000 \\
    rollout_manager.base_url=\"http://localhost:$PORT\" \\
    2>&1 | tee $SCRIPT_DIR/logs/$EXPERIMENT_NAME.log" C-m

echo "-----------------------------------------"
echo "SPOC GRPO Training Configuration Summary:"
echo "Port: $PORT"
echo "CUDA Devices: $CUDA_DEVICES"
#
echo "Server Session: $SERVER_SESSION"
echo "Training Session: $TRAIN_SESSION"
echo "Environment: SPOC (Stretch robot manipulation)"
echo "Task Type: Fetch"
echo "Prompt Format: grounding_worldmodeling"
echo "-----------------------------------------"
echo "LOG FILE LOCATIONS:"
echo "Server log: $SCRIPT_DIR/logs/server.log"
echo "Training log: $SCRIPT_DIR/logs/$EXPERIMENT_NAME.log"
echo ""
echo "To monitor logs in real-time:"
echo "Server: tail -f $SCRIPT_DIR/logs/server.log"
echo "Training: tail -f $SCRIPT_DIR/logs/$EXPERIMENT_NAME.log"
echo "-----------------------------------------"
echo "To attach to server session: tmux attach-session -t $SERVER_SESSION"
echo "To attach to training session: tmux attach-session -t $TRAIN_SESSION"
echo "NOTE: The sessions will remain active. To detach from a session use Ctrl+B followed by D"
echo ""
echo "SPOC-specific adjustments made for 4x A100:"
echo "- Model: Qwen2.5-VL-3B-Instruct (latest 3B multimodal model)"
echo "- Quad GPU configuration (n_gpus_per_node=4)"
echo "- Tensor model parallel size: 4 (quad GPU)"
echo "- GPU memory utilization: 0.25 (optimized for 4 GPU setup)"
echo "- Train batch size: 4 (scaled for quad GPU)"
echo "- PPO mini batch size: 4 (scaled for quad GPU)"
echo "- Trajectory count: 1 (每 GPU 单环境，减小验证并发)"
echo "- Max trajectory length: 1200 (足以覆盖长 prompt)"
echo "- Max response length: 200 (rollout 再限 256，上限靠 max_response_length 控制)"
echo "- Max num seqs: 8, Max batched tokens: 3600 (scaled for 4 GPU)"
echo "- Enforce eager mode: True (no CUDA graphs)"
echo "- Free cache engine: True (release memory)"
echo "- FSDP parameter/optimizer offloading enabled for memory efficiency"
echo "- Ray resource management optimized for container environment"
echo "- PyTorch CUDA allocator optimized (max_split_size_mb=128)"
echo "- Validation frequency延长至 200 step 以减少验证开销"
echo "- Increased timeout to 600s for Stretch robot interactions" 