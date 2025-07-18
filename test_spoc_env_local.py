#!/usr/bin/env python3
"""
本地单元测试 SPOC 环境创建问题
测试不同的配置组合，打印详细的调试信息
"""
import os
import sys
import subprocess
import json

# 设置 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("SPOC 环境本地测试")
print("=" * 80)

# 1. 检查并打印当前环境变量
print("\n1. 检查环境变量:")
important_vars = [
    'DISPLAY', 'MESA_GL_VERSION_OVERRIDE', 'LIBGL_ALWAYS_SOFTWARE',
    'EGL_PLATFORM', 'GALLIUM_DRIVER', 'SPOC_DATA_PATH', 
    'CUDA_VISIBLE_DEVICES', 'PATH'
]
for var in important_vars:
    value = os.environ.get(var, 'NOT SET')
    print(f"  {var}: {value}")

# 2. 检查 fake vulkaninfo
print("\n2. 检查 vulkaninfo:")
try:
    result = subprocess.run(['which', 'vulkaninfo'], capture_output=True, text=True)
    print(f"  vulkaninfo 路径: {result.stdout.strip()}")
    
    result = subprocess.run(['vulkaninfo'], capture_output=True, text=True)
    print(f"  vulkaninfo 输出: {result.stdout.strip()}")
    print(f"  vulkaninfo 返回码: {result.returncode}")
except Exception as e:
    print(f"  vulkaninfo 错误: {e}")

# 3. 检查 SPOC 数据路径
print("\n3. 检查 SPOC 数据:")
spoc_data_path = os.environ.get('SPOC_DATA_PATH', '/home/jiajunliu/spoc_data/fifteen')
print(f"  数据路径: {spoc_data_path}")
print(f"  路径存在: {os.path.exists(spoc_data_path)}")
if os.path.exists(spoc_data_path):
    files = os.listdir(spoc_data_path)[:5]  # 只显示前5个文件
    print(f"  数据文件: {files}")

# 4. 尝试导入 SPOC 环境
print("\n4. 尝试导入 SPOC 环境:")
try:
    from vagen.env.spoc.env import SpocEnv, SpocEnvConfig
    print("  ✓ 成功导入 SpocEnv 和 SpocEnvConfig")
except ImportError as e:
    print(f"  ✗ 导入失败: {e}")
    sys.exit(1)

# 5. 设置测试环境变量
print("\n5. 设置测试环境变量:")
test_envs = {
    'MESA_GL_VERSION_OVERRIDE': '3.3',
    'LIBGL_ALWAYS_SOFTWARE': '1',
    'EGL_PLATFORM': 'surfaceless',
    'GALLIUM_DRIVER': 'llvmpipe',
    'SPOC_DATA_PATH': '/home/jiajunliu/spoc_data/fifteen',
}

for key, value in test_envs.items():
    os.environ[key] = value
    print(f"  设置 {key}={value}")

# 创建 fake vulkaninfo
home_bin = os.path.expanduser("~/bin")
os.makedirs(home_bin, exist_ok=True)
vulkaninfo_path = os.path.join(home_bin, "vulkaninfo")
with open(vulkaninfo_path, 'w') as f:
    f.write('#!/bin/bash\necho "Fake vulkaninfo"\nexit 0\n')
os.chmod(vulkaninfo_path, 0o755)
os.environ['PATH'] = f"{home_bin}:{os.environ['PATH']}"
print(f"  创建 fake vulkaninfo: {vulkaninfo_path}")

# 6. 测试不同的 DISPLAY 配置
print("\n6. 测试环境创建:")

configs = [
    ("unset DISPLAY", lambda: os.environ.pop('DISPLAY', None)),
    ("DISPLAY=:0", lambda: os.environ.update({'DISPLAY': ':0'})),
    ("DISPLAY=:99", lambda: os.environ.update({'DISPLAY': ':99'})),
]

for config_name, setup_func in configs:
    print(f"\n  测试配置: {config_name}")
    setup_func()
    
    try:
        # 创建配置
        config = SpocEnvConfig(
            data_path="/home/jiajunliu/spoc_data/fifteen",
            task_type="FetchType",
            chores_split="train",
            prompt_format="grounding_worldmodeling",
            max_actions_per_step=3,
            action_sep=",",
            resolution=224,
            fov=90,
            step_length=0.2,
            gpu_device=0
        )
        print(f"    配置创建成功")
        
        # 创建环境
        print(f"    开始创建 SPOC 环境...")
        env = SpocEnv(config)
        print(f"    ✓ 环境创建成功!")
        
        # 测试 reset
        print(f"    测试 reset...")
        obs, info = env.reset(seed=42)
        print(f"    ✓ Reset 成功!")
        print(f"    观察空间: {type(obs)}")
        if 'images' in obs:
            print(f"    图像数量: {len(obs['images'])}")
        
        # 测试 system prompt
        system_prompt = env.system_prompt
        print(f"    系统提示长度: {len(system_prompt)}")
        
        # 关闭环境
        env.close()
        print(f"    ✓ 环境关闭成功")
        
        # 找到成功的配置就退出
        print(f"\n  ✓✓✓ 成功配置: {config_name}")
        break
        
    except Exception as e:
        print(f"    ✗ 错误: {type(e).__name__}: {str(e)}")
        # 打印更详细的错误信息
        import traceback
        traceback.print_exc()

# 7. 测试 AI2-THOR 直接初始化
print("\n7. 直接测试 AI2-THOR:")
try:
    from ai2thor.controller import Controller
    
    platforms = ["CloudRendering", "Linux64"]
    for platform in platforms:
        print(f"\n  测试平台: {platform}")
        try:
            controller = Controller(
                platform=platform,
                scene="FloorPlan1_physics",
                width=224,
                height=224,
                fieldOfView=90,
                agentMode="stretch",
                renderDepthImage=False,
                renderInstanceSegmentation=True,
                visibilityDistance=30
            )
            print(f"    ✓ {platform} 初始化成功!")
            controller.stop()
            break
        except Exception as e:
            print(f"    ✗ {platform} 失败: {e}")

except ImportError:
    print("  ✗ AI2-THOR 未安装")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)