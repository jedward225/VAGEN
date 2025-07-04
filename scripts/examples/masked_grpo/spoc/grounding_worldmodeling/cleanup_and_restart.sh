 

# 清理脚本 - 用于重新运行SPOC训练
echo "正在清理之前的运行..."

# 获取脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPERIMENT_NAME=$(echo $SCRIPT_DIR | rev | cut -d'/' -f1-3 | rev | tr '/' '-')

# 清理tmux会话
echo "清理tmux会话..."
tmux kill-session -t spoc-server0 2>/dev/null || true
tmux kill-session -t spoc-server1 2>/dev/null || true
tmux kill-session -t spoc-server2 2>/dev/null || true
tmux kill-session -t spoc-train0 2>/dev/null || true
tmux kill-session -t spoc-train1 2>/dev/null || true
tmux kill-session -t spoc-train2 2>/dev/null || true

# 清理数据文件
echo "清理数据文件..."
rm -rf "data/$EXPERIMENT_NAME" 2>/dev/null || true

# 清理日志文件
echo "清理日志文件..."
rm -f "$EXPERIMENT_NAME.log" 2>/dev/null || true

# 清理Ray相关进程
echo "清理Ray进程..."
ray stop 2>/dev/null || true
sleep 2

# 清理CUDA进程（如果有残留）
echo "检查CUDA进程..."
nvidia-smi | grep python && echo "检测到CUDA进程，可能需要手动清理" || echo "未检测到残留的CUDA进程"

echo "清理完成！"
echo "现在可以重新运行: bash run_tmux.sh"
echo "或者可以手动运行每个步骤进行调试" 
 