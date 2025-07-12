#!/bin/bash

echo "=== SPOC训练快速修复和重启脚本 ==="

# 1. 杀掉所有相关进程
echo "1. 清理进程..."
pkill -f "vagen.server.server" || true
pkill -f "vagen.trainer.main_ppo" || true

# 2. 清理tmux会话
echo "2. 清理tmux会话..."
tmux kill-session -t spoc-server0 2>/dev/null || true
tmux kill-session -t spoc-server1 2>/dev/null || true
tmux kill-session -t spoc-train0 2>/dev/null || true
tmux kill-session -t spoc-train1 2>/dev/null || true

# 3. 清理Ray
echo "3. 清理Ray..."
ray stop 2>/dev/null || true
sleep 2

# 4. 清理端口
echo "4. 检查端口占用..."
netstat -tulpn | grep :5000 || echo "端口5000空闲"

# 5. 等待一下
echo "5. 等待进程完全清理..."
sleep 5

# 6. 重新运行
echo "6. 重新启动训练..."
cd "$(dirname "$0")"
bash run_tmux.sh

echo "=== 重启完成 ===" 