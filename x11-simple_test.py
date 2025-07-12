#!/usr/bin/env python3
"""
简单的AI2-THOR测试脚本
测试X11转发和基本环境创建
"""

import os
import sys
import traceback

def test_display():
    """测试显示设置"""
    print("=== 测试显示设置 ===")
    display = os.environ.get('DISPLAY')
    print(f"DISPLAY环境变量: {display}")
    return display is not None

def test_ai2thor():
    """测试AI2-THOR基本功能"""
    print("\n=== 测试AI2-THOR ===")
    
    try:
        import ai2thor.controller
        print("✓ AI2-THOR导入成功")
    except ImportError as e:
        print(f"✗ AI2-THOR导入失败: {e}")
        return False
    
    # 检测显示环境
    display_available = os.environ.get('DISPLAY') is not None
    
    # 配置控制器
    config = {
        "agentMode": "default",
        "gridSize": 0.1,
        "width": 100,
        "height": 80,
        "fieldOfView": 90,
        "platform": "Linux64" if display_available else "CloudRendering",
        "headless": False,
        "gpu_device": 0,
        "server_timeout": 300,
        "quality": "Low",
    }
    
    print(f"使用平台: {config['platform']}")
    print(f"无头模式: {config['headless']}")
    print(f"显示可用: {display_available}")
    
    try:
        print("正在创建AI2-THOR控制器...")
        controller = ai2thor.controller.Controller(**config)
        print("✓ AI2-THOR控制器创建成功")
        
        try:
            print("正在重置环境...")
            event = controller.reset()
            print("✓ 环境重置成功")
            
            if event.metadata:
                print(f"  - 场景: {event.metadata.get('sceneName', 'unknown')}")
                if 'agent' in event.metadata:
                    agent_pos = event.metadata['agent'].get('position', {})
                    print(f"  - 代理位置: x={agent_pos.get('x', 0):.2f}, z={agent_pos.get('z', 0):.2f}")
                
                if hasattr(event, 'frame') and event.frame is not None:
                    print(f"  - 图像帧: {event.frame.shape}")
                
            print("正在停止控制器...")
            controller.stop()
            print("✓ 控制器停止成功")
            return True
            
        except Exception as e:
            print(f"✗ 控制器操作失败: {e}")
            traceback.print_exc()
            try:
                controller.stop()
            except:
                pass
            return False
            
    except Exception as e:
        print(f"✗ AI2-THOR控制器创建失败: {e}")
        traceback.print_exc()
        return False

def test_x11_forwarding():
    """测试X11转发"""
    print("\n=== 测试X11转发 ===")
    
    display = os.environ.get('DISPLAY')
    if not display:
        print("✗ DISPLAY环境变量未设置")
        return False
    
    try:
        # 尝试运行一个简单的X11命令
        import subprocess
        result = subprocess.run(['xset', 'q'], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ X11转发正常工作")
            print(f"  - X服务器信息: {result.stdout.split('X.Org')[0].strip()}")
            return True
        else:
            print("✗ X11转发不工作")
            print(f"  - 错误: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("⚠ xset命令未找到，尝试其他方法")
        
        # 尝试通过Python检测
        try:
            import tkinter
            root = tkinter.Tk()
            root.withdraw()  # 隐藏窗口
            root.destroy()
            print("✓ X11转发工作（通过tkinter测试）")
            return True
        except Exception as e:
            print(f"✗ X11转发测试失败: {e}")
            return False
            
    except Exception as e:
        print(f"✗ X11转发测试失败: {e}")
        return False

def main():
    """主函数"""
    print("开始简单的AI2-THOR和X11测试...")
    
    tests = [
        ("显示设置", test_display),
        ("X11转发", test_x11_forwarding),
        ("AI2-THOR", test_ai2thor),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} 测试异常: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    print(f"\n{'='*50}")
    print("=== 测试结果总结 ===")
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 所有测试通过！AI2-THOR应该可以正常工作。")
        print("现在可以尝试运行SPOC训练了。")
    else:
        print("\n❌ 部分测试失败，需要先修复相关问题。")
        
        # 提供建议
        failed_tests = [name for name, result in results if not result]
        if "显示设置" in failed_tests:
            print("\n💡 建议：")
            print("  1. 确保SSH连接时使用了 -X 或 -Y 选项")
            print("  2. 检查本地X11服务器是否运行")
            print("  3. 尝试: export DISPLAY=:0.0")
        
        if "X11转发" in failed_tests:
            print("\n💡 建议：")
            print("  1. 重新连接SSH: ssh -X username@server")
            print("  2. 检查本地防火墙设置")
            print("  3. 尝试安装: sudo apt install x11-utils")
        
        if "AI2-THOR" in failed_tests:
            print("\n💡 建议：")
            print("  1. 检查vulkan驱动是否正确安装")
            print("  2. 尝试: sudo apt install vulkan-tools libvulkan1")
            print("  3. 检查GPU驱动是否正确")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 