#!/bin/bash

# SPOC GRPO训练监控脚本
# 使用方法: ./monitor.sh

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 清屏
clear

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    SPOC GRPO训练监控面板${NC}"
echo -e "${BLUE}========================================${NC}"

# 函数：检查tmux会话状态
check_tmux_sessions() {
    echo -e "\n${YELLOW}📋 Tmux会话状态:${NC}"
    if tmux list-sessions 2>/dev/null | grep -E "(spoc-server|spoc-train)"; then
        echo -e "${GREEN}✅ 发现SPOC训练会话${NC}"
        
        # 检查服务器会话
        if tmux list-sessions 2>/dev/null | grep "spoc-server" > /dev/null; then
            echo -e "${GREEN}🖥️  服务器会话: 运行中${NC}"
        else
            echo -e "${RED}❌ 服务器会话: 未找到${NC}"
        fi
        
        # 检查训练会话
        if tmux list-sessions 2>/dev/null | grep "spoc-train" > /dev/null; then
            echo -e "${GREEN}🚀 训练会话: 运行中${NC}"
        else
            echo -e "${RED}❌ 训练会话: 未找到${NC}"
        fi
    else
        echo -e "${RED}❌ 未找到SPOC训练会话${NC}"
    fi
}

# 函数：检查GPU使用情况
check_gpu_usage() {
    echo -e "\n${YELLOW}🔥 GPU使用情况:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while read line; do
            echo -e "${GREEN}GPU $line${NC}"
        done
    else
        echo -e "${RED}❌ nvidia-smi未找到${NC}"
    fi
}

# 函数：检查日志文件
check_log_files() {
    echo -e "\n${YELLOW}📄 日志文件:${NC}"
    
    # 查找训练日志
    if ls masked_grpo-spoc-grounding_worldmodeling*.log 2>/dev/null; then
        latest_log=$(ls -t masked_grpo-spoc-grounding_worldmodeling*.log | head -1)
        echo -e "${GREEN}📝 最新日志: $latest_log${NC}"
        
        # 显示最后10行
        echo -e "\n${YELLOW}📊 最近训练日志:${NC}"
        tail -10 "$latest_log" 2>/dev/null || echo -e "${RED}❌ 无法读取日志文件${NC}"
    else
        echo -e "${RED}❌ 未找到训练日志文件${NC}"
    fi
}

# 函数：检查端口占用
check_port_usage() {
    echo -e "\n${YELLOW}🔌 端口使用情况:${NC}"
    
    # 检查5000端口
    if netstat -tlnp 2>/dev/null | grep ":5000 " > /dev/null; then
        echo -e "${GREEN}✅ 端口5000: 被占用（服务器运行中）${NC}"
    else
        echo -e "${RED}❌ 端口5000: 未被占用${NC}"
    fi
}

# 函数：检查数据文件
check_data_files() {
    echo -e "\n${YELLOW}📂 数据文件:${NC}"
    
    if [ -d "data" ]; then
        echo -e "${GREEN}📁 数据目录存在${NC}"
        
        # 检查训练数据
        if ls data/*/train.parquet 2>/dev/null; then
            for file in data/*/train.parquet; do
                size=$(du -h "$file" | cut -f1)
                echo -e "${GREEN}📊 训练数据: $file ($size)${NC}"
            done
        else
            echo -e "${RED}❌ 训练数据文件未找到${NC}"
        fi
        
        # 检查测试数据
        if ls data/*/test.parquet 2>/dev/null; then
            for file in data/*/test.parquet; do
                size=$(du -h "$file" | cut -f1)
                echo -e "${GREEN}📊 测试数据: $file ($size)${NC}"
            done
        else
            echo -e "${RED}❌ 测试数据文件未找到${NC}"
        fi
    else
        echo -e "${RED}❌ 数据目录不存在${NC}"
    fi
}

# 函数：提供快速操作选项
show_quick_actions() {
    echo -e "\n${YELLOW}🚀 快速操作:${NC}"
    echo -e "${BLUE}1. 查看服务器状态:${NC} tmux attach-session -t \$(tmux list-sessions | grep spoc-server | cut -d: -f1)"
    echo -e "${BLUE}2. 查看训练状态:${NC} tmux attach-session -t \$(tmux list-sessions | grep spoc-train | cut -d: -f1)"
    echo -e "${BLUE}3. 查看实时日志:${NC} tail -f masked_grpo-spoc-grounding_worldmodeling*.log"
    echo -e "${BLUE}4. 查看GPU监控:${NC} watch -n 1 nvidia-smi"
    echo -e "${BLUE}5. 重新运行监控:${NC} ./monitor.sh"
    echo -e "${BLUE}6. 停止所有会话:${NC} tmux kill-session -t \$(tmux list-sessions | grep spoc | cut -d: -f1)"
}

# 主函数：执行所有检查
main() {
    check_tmux_sessions
    check_gpu_usage
    check_port_usage
    check_data_files
    check_log_files
    show_quick_actions
    
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}监控完成! 使用 './monitor.sh' 重新运行${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# 如果脚本带参数，执行相应操作
case "${1:-monitor}" in
    "server")
        echo "连接到服务器会话..."
        tmux attach-session -t $(tmux list-sessions | grep spoc-server | cut -d: -f1)
        ;;
    "train")
        echo "连接到训练会话..."
        tmux attach-session -t $(tmux list-sessions | grep spoc-train | cut -d: -f1)
        ;;
    "log")
        echo "查看实时日志..."
        tail -f masked_grpo-spoc-grounding_worldmodeling*.log
        ;;
    "gpu")
        echo "查看GPU监控..."
        watch -n 1 nvidia-smi
        ;;
    *)
        main
        ;;
esac