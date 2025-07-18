#!/usr/bin/env python3
"""
测试 SPOC 服务器环境创建
找出为什么在 run_tmux.sh 中会失败
"""
import os
import sys
import subprocess
import time
import requests
import json
import signal

# 设置环境变量
print("设置环境变量...")
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['EGL_PLATFORM'] = 'surfaceless'
os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
os.environ['SPOC_DATA_PATH'] = '/home/jiajunliu/spoc_data/fifteen'
os.environ['PYTHONPATH'] = f"/home/jiajunliu/VAGEN:{os.environ.get('PYTHONPATH', '')}"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# 创建 fake vulkaninfo
home_bin = os.path.expanduser("~/bin")
os.makedirs(home_bin, exist_ok=True)
vulkaninfo_path = os.path.join(home_bin, "vulkaninfo")
with open(vulkaninfo_path, 'w') as f:
    f.write('#!/bin/bash\necho "Fake vulkaninfo"\nexit 0\n')
os.chmod(vulkaninfo_path, 0o755)
os.environ['PATH'] = f"{home_bin}:{os.environ['PATH']}"

# 测试不同的服务器启动方式
test_configs = [
    {
        "name": "直接启动（像 tmux 一样）",
        "cmd": ["python", "-m", "vagen.server.server", "server.port=5002"],
        "unset_display": True
    },
    {
        "name": "使用 xvfb-run",
        "cmd": ["xvfb-run", "-a", "-s", "-screen 0 1024x768x24", 
                "python", "-m", "vagen.server.server", "server.port=5003"],
        "unset_display": False
    }
]

for config in test_configs:
    print(f"\n{'='*60}")
    print(f"测试: {config['name']}")
    print(f"{'='*60}")
    
    # 设置 DISPLAY
    if config['unset_display']:
        if 'DISPLAY' in os.environ:
            del os.environ['DISPLAY']
            print("已 unset DISPLAY")
    
    # 启动服务器
    print(f"启动命令: {' '.join(config['cmd'])}")
    server_process = subprocess.Popen(
        config['cmd'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    print(f"服务器 PID: {server_process.pid}")
    print("等待服务器启动...")
    
    # 收集服务器输出
    server_output = []
    start_time = time.time()
    while time.time() - start_time < 15:
        import select
        ready, _, _ = select.select([server_process.stdout], [], [], 0.1)
        if ready:
            line = server_process.stdout.readline()
            if line:
                server_output.append(line.strip())
                print(f"[SERVER] {line.strip()}")
        time.sleep(0.1)
    
    # 获取端口
    port = config['cmd'][-1].split('=')[-1]
    base_url = f"http://localhost:{port}"
    
    # 测试健康检查
    print(f"\n测试健康检查...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"健康检查: {response.status_code}")
        if response.status_code == 200:
            print(f"响应: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"健康检查失败: {e}")
    
    # 测试环境创建
    print(f"\n测试创建环境...")
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
        response = requests.post(
            f"{base_url}/environments", 
            json={"ids2configs": ids2configs}, 
            timeout=60
        )
        print(f"环境创建状态: {response.status_code}")
        print(f"响应: {response.text[:500]}")  # 只打印前500字符
        
        if response.status_code == 200:
            # 测试 reset
            print(f"\n测试 reset val1...")
            reset_response = requests.post(
                f"{base_url}/reset/val1", 
                json={"seed": 42},
                timeout=10
            )
            print(f"Reset 状态: {reset_response.status_code}")
            if reset_response.status_code == 200:
                print("✓ 环境创建成功！")
            else:
                print(f"Reset 错误: {reset_response.text}")
    except Exception as e:
        print(f"环境创建失败: {e}")
    
    # 打印服务器最后的输出
    print(f"\n收集剩余服务器输出...")
    end_time = time.time() + 2
    while time.time() < end_time:
        ready, _, _ = select.select([server_process.stdout], [], [], 0.1)
        if ready:
            line = server_process.stdout.readline()
            if line:
                print(f"[SERVER] {line.strip()}")
    
    # 关闭服务器
    print(f"\n关闭服务器...")
    server_process.terminate()
    server_process.wait()
    
    print(f"\n分析: 查找关键错误信息...")
    for line in server_output:
        if "error" in line.lower() or "failed" in line.lower() or "vulkan" in line.lower():
            print(f"  ! {line}")

print("\n" + "="*60)
print("测试完成")
print("="*60)