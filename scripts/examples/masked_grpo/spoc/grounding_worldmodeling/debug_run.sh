 

# 调试脚本 - 逐步运行SPOC训练
echo "SPOC训练调试脚本"
echo "===================="

# 获取脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENT_NAME=$(echo $SCRIPT_DIR | rev | cut -d'/' -f1-3 | rev | tr '/' '-')

# 默认配置
PORT=5000
CUDA_DEVICES="0,1"

echo "实验名称: $EXPERIMENT_NAME"
echo "端口: $PORT"
echo "CUDA设备: $CUDA_DEVICES"

# 创建数据目录
echo "创建数据目录..."
mkdir -p "data/$EXPERIMENT_NAME"

# 激活环境
echo "激活conda环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vagen

# 设置环境变量
echo "设置环境变量..."
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_DISABLE_RESOURCE_AUTOSCALING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export MASTER_PORT=29500

echo "步骤1: 创建数据集"
echo "运行命令: python -m vagen.env.create_dataset --yaml_path env_config.yaml --train_path data/$EXPERIMENT_NAME/train.parquet --test_path data/$EXPERIMENT_NAME/test.parquet"
python -m vagen.env.create_dataset \
    --yaml_path "$SCRIPT_DIR/env_config.yaml" \
    --train_path "data/$EXPERIMENT_NAME/train.parquet" \
    --test_path "data/$EXPERIMENT_NAME/test.parquet"

if [ $? -eq 0 ]; then
    echo "✓ 数据集创建成功"
else
    echo "✗ 数据集创建失败"
    exit 1
fi

echo ""
echo "步骤2: 启动服务器"
echo "在另一个终端运行以下命令:"
echo "conda activate vagen"
echo "export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES"
echo "export VLLM_ATTENTION_BACKEND=XFORMERS"
echo "export PYTHONHASHSEED=0"
echo "export RAY_DISABLE_DOCKER_CPU_WARNING=1"
echo "export RAY_DISABLE_RESOURCE_AUTOSCALING=1"
echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128"
echo "python -m vagen.server.server server.port=$PORT"
echo ""
echo "按Enter键继续（确保服务器已启动）..."
read

echo ""
echo "步骤3: 启动训练"
echo "运行训练脚本..."
cd "$SCRIPT_DIR"
python3 -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.high_level_gamma=1.0 \
    data.train_files=data/$EXPERIMENT_NAME/train.parquet \
    data.val_files=data/$EXPERIMENT_NAME/test.parquet \
    data.train_batch_size=2 \
    data.max_prompt_length=1024 \
    data.max_response_length=200 \
    data.max_trajectory_length=512 \
    data.image_key=images \
    data.truncation=left \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.temperature=0.7 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.ppo_mini_batch_size=2 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger='[console,wandb]' \
    trainer.project_name='vagen_spoc' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=25 \
    trainer.total_training_steps=100 \
    rollout_manager.max_turns=5 \
    rollout_manager.window_size=8 \
    rollout_manager.use_multi_turn_reward=False \
    rollout_manager.use_loss_mask=True \
    rollout_manager.use_gae_mask=True \
    rollout_manager.n_gpus_per_node=2 \
    trainer.val_before_train=True \
    trainer.val_generations_to_log_to_wandb=8 \
    rollout_manager.n_trajectory=2 \
    rollout_manager.use_service=True \
    rollout_manager.timeout=600 \
    rollout_manager.base_url="http://localhost:$PORT" \
    2>&1 | tee $EXPERIMENT_NAME.log

echo "训练完成！" 
 