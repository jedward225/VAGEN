#!/usr/bin/env python3
"""
测试AI2-THOR环境创建的脚本
用于诊断环境创建问题
"""

import os
import sys
import traceback

# 直接导入避免通过 vagen.env.__init__.py 导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vagen'))

try:
    from vagen.env.spoc.env_config import SpocEnvConfig
    from vagen.env.spoc.env import SpocEnv
    IMPORTS_OK = True
except ImportError as e:
    print(f"导入错误: {e}")
    IMPORTS_OK = False
    SpocEnvConfig = None
    SpocEnv = None

def test_display_setup():
    """测试显示设置"""
    print("=== 测试显示设置 ===")
    display = os.environ.get('DISPLAY')
    print(f"DISPLAY环境变量: {display}")
    
    if display:
        print("✓ DISPLAY环境变量已设置")
    else:
        print("✗ DISPLAY环境变量未设置")
    
    return display is not None

def test_basic_import():
    """测试基本导入"""
    print("\n=== 测试基本导入 ===")
    try:
        import ai2thor.controller
        print("✓ ai2thor.controller 导入成功")
        return True
    except ImportError as e:
        print(f"✗ ai2thor.controller 导入失败: {e}")
        return False

def test_env_config():
    """测试环境配置"""
    print("\n=== 测试环境配置 ===")
    if not IMPORTS_OK:
        print("✗ 导入失败，跳过环境配置测试")
        return False
    try:
        config = SpocEnvConfig()
        print("✓ SpocEnvConfig 创建成功")
        print(f"  - 分辨率: {config.resolution}")
        print(f"  - FOV: {config.fov}")
        print(f"  - 任务类型: {config.task_type}")
        return True
    except Exception as e:
        print(f"✗ SpocEnvConfig 创建失败: {e}")
        return False

def test_thor_controller():
    """测试AI2-THOR控制器创建"""
    print("\n=== 测试AI2-THOR控制器 ===")
    
    # 检测显示环境
    display_available = os.environ.get('DISPLAY') is not None
    print(f"显示环境检测: {'可用' if display_available else '不可用'}")
    
    # 基本配置
    thor_config = {
        "agentMode": "default",
        "gridSize": 0.1,
        "visibilityDistance": 10,
        "renderDepthImage": False,
        "renderInstanceSegmentation": False,
        "width": 224,
        "height": 224,
        "fieldOfView": 90,
        "platform": "CloudRendering" if not display_available else "Linux64",
        "headless": False,
        "gpu_device": 0,
        "server_timeout": 300,
        "server_start_timeout": 300,
        "quality": "Low",
    }
    
    print(f"使用平台: {thor_config['platform']}")
    print(f"无头模式: {thor_config['headless']}")
    
    try:
        import ai2thor.controller
        controller = ai2thor.controller.Controller(**thor_config)
        print("✓ AI2-THOR控制器创建成功")
        
        # 尝试简单的操作
        try:
            event = controller.reset()
            print("✓ 控制器重置成功")
            print(f"  - 场景: {event.metadata.get('sceneName', 'unknown')}")
            print(f"  - 代理位置: {event.metadata.get('agent', {}).get('position', 'unknown')}")
            
            controller.stop()
            print("✓ 控制器停止成功")
            return True
        except Exception as e:
            print(f"✗ 控制器操作失败: {e}")
            try:
                controller.stop()
            except:
                pass
            return False
            
    except Exception as e:
        print(f"✗ AI2-THOR控制器创建失败: {e}")
        traceback.print_exc()
        return False

def test_spoc_env():
    """测试SPOC环境创建"""
    print("\n=== 测试SPOC环境创建 ===")
    
    if not IMPORTS_OK:
        print("✗ 导入失败，跳过SPOC环境测试")
        return False
    
    try:
        config = SpocEnvConfig()
        env = SpocEnv(config)
        print("✓ SPOC环境创建成功")
        
        try:
            obs, info = env.reset(seed=0)
            print("✓ 环境重置成功")
            print(f"  - obs_str 存在: {'obs_str' in obs}")
            print(f"  - multi_modal_data 存在: {'multi_modal_data' in obs}")
            
            if 'obs_str' in obs:
                print(f"  - obs_str 长度: {len(obs['obs_str'])}")
            
            env.close()
            print("✓ 环境关闭成功")
            return True
            
        except Exception as e:
            print(f"✗ 环境操作失败: {e}")
            traceback.print_exc()
            try:
                env.close()
            except:
                pass
            return False
            
    except Exception as e:
        print(f"✗ SPOC环境创建失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始AI2-THOR环境创建测试...")
    
    tests = [
        ("显示设置", test_display_setup),
        ("基本导入", test_basic_import),
        ("环境配置", test_env_config),
        ("AI2-THOR控制器", test_thor_controller),
        ("SPOC环境", test_spoc_env),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} 测试出现异常: {e}")
            results.append((name, False))
    
    print("\n=== 测试结果总结 ===")
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 所有测试通过！环境创建应该可以正常工作。")
    else:
        print("\n❌ 部分测试失败，需要修复相关问题。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 