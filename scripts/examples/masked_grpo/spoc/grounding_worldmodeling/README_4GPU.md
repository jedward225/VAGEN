# SPOC 4-GPU 配置说明

本文档说明了如何修改SPOC环境以支持4卡运行。

## 主要修改

### 1. 运行脚本修改 (`run_tmux.sh`)

**修改内容：**
- 默认CUDA设备从 `0,1` 改为 `0,1,2,3`
- 训练批次大小从 2 增加到 4
- PPO mini batch size 从 2 增加到 4
- Tensor model parallel size 从 2 增加到 4
- GPU内存利用率从 0.34 降低到 0.25
- Max num seqs 从 4 增加到 8
- Max batched tokens 从 1800 增加到 3600
- Trainer和rollout manager的GPU数量从 2 增加到 4

### 2. 服务配置修改 (`service_config.py`)

**修改内容：**
- 默认设备列表从 `[0]` 改为 `[0, 1, 2, 3]`

### 3. 环境配置修改 (`env_config.yaml`)

**修改内容：**
- 添加了4个环境配置 (env1-env4)
- 每个环境配置对应一个GPU设备 (gpu_device: 0, 1, 2, 3)

### 4. 新增4卡优化脚本 (`run_tmux_4gpu.sh`)

**特点：**
- 专门为4卡运行优化
- 更高的批次大小 (train_batch_size=8)
- 更高的PPO mini batch size (8)
- 更高的micro batch size per GPU (2)
- 更高的max num seqs (16)
- 更高的max batched tokens (7200)
- 更低的GPU内存利用率 (0.20)
- 更长的服务器启动等待时间 (20s)

## 使用方法

### 方法1：使用修改后的原脚本
```bash
cd VAGEN/scripts/examples/masked_grpo/spoc/grounding_worldmodeling/
./run_tmux.sh
```

### 方法2：使用4卡优化脚本
```bash
cd VAGEN/scripts/examples/masked_grpo/spoc/grounding_worldmodeling/
./run_tmux_4gpu.sh
```

## 配置参数对比

| 参数 | 2卡配置 | 4卡配置 | 4卡优化配置 |
|------|---------|---------|-------------|
| CUDA Devices | 0,1 | 0,1,2,3 | 0,1,2,3 |
| Train Batch Size | 2 | 4 | 8 |
| PPO Mini Batch Size | 2 | 4 | 8 |
| PPO Micro Batch Size per GPU | 1 | 1 | 2 |
| Tensor Model Parallel Size | 2 | 4 | 4 |
| GPU Memory Utilization | 0.34 | 0.25 | 0.20 |
| Max Num Seqs | 4 | 8 | 16 |
| Max Batched Tokens | 1800 | 3600 | 7200 |
| N GPUs Per Node | 2 | 4 | 4 |

## 性能优化建议

1. **内存管理**：4卡配置使用更低的GPU内存利用率以避免OOM
2. **批次大小**：根据GPU数量线性增加批次大小
3. **并行度**：增加micro batch size per GPU以提高并行效率
4. **启动时间**：增加服务器启动等待时间以适应4卡初始化

## 故障排除

### 常见问题

1. **CUDA OOM错误**
   - 降低GPU内存利用率
   - 减少批次大小
   - 启用FSDP参数卸载

2. **初始化超时**
   - 增加服务器启动等待时间
   - 检查GPU设备可用性

3. **NCCL通信错误**
   - 确保NCCL_P2P_DISABLE=1
   - 确保NCCL_IB_DISABLE=1

### 监控命令

```bash
# 查看GPU使用情况
nvidia-smi

# 查看tmux会话
tmux list-sessions

# 查看训练日志
tail -f EXPERIMENT_NAME.log
```

## 注意事项

1. 确保系统有足够的GPU内存 (建议每卡至少24GB)
2. 确保CUDA版本兼容性
3. 监控GPU温度和功耗
4. 根据实际硬件调整批次大小和内存利用率 