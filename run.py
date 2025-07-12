import ai2thor.controller
import os

# --- 关键参数设定 ---

# 1. 精确指定可执行文件路径，这是我们调试成功的关键发现
COMMIT_ID = "f0825767cd50d69f666c7f282e54abfe58f1e917"
EXECUTABLE_NAME = f"thor-Linux64-{COMMIT_ID}"
EXECUTABLE_PATH = os.path.expanduser(f"~/.ai2thor/releases/{EXECUTABLE_NAME}/{EXECUTABLE_NAME}")

# 2. 将超时时间大幅延长至300秒，以应对服务器可能存在的启动缓慢问题
ACTION_TIMEOUT = 300.0

# --- 脚本开始 ---

print("--- 最终解决方案脚本 ---")
print(f"🔩 强制使用执行文件: {EXECUTABLE_PATH}")

if not os.path.exists(EXECUTABLE_PATH):
    print(f"❌ 致命错误: 指定的执行文件路径不存在！")
    exit()

print(f"⏳ 正在启动控制器 (超时设置为 {ACTION_TIMEOUT} 秒)...")

try:
    # 核心修正：
    # 1. 使用 local_executable_path
    # 2. 移除 headless=True 参数，因为 xvfb-run 会处理图形环境
    # 3. 增加 timeout 参数
    controller = ai2thor.controller.Controller(
        local_executable_path=EXECUTABLE_PATH,
        timeout=ACTION_TIMEOUT,
        scene="FloorPlan1",
        gridSize=0.25,
        width=300,
        height=300
    )

    print("\n\n" + "="*50)
    print("✅✅✅ 恭喜！控制器已成功启动！ ✅✅✅")
    print("="*50 + "\n")
    
    event = controller.last_event
    position = event.metadata['agent']['position']
    print(f"🤖 智能体起始位置: {position}")

    print("🏃 执行动作: 'MoveAhead'")
    event = controller.step(action="MoveAhead")
    position = event.metadata['agent']['position']
    print(f"🤖 智能体新位置: {position}")

    print("\n\n🎉🎉🎉 程序执行完毕，环境正常工作！🎉🎉🎉")

except Exception as e:
    print("\n\n" + "="*50)
    print("❌❌❌ 控制器启动或运行失败！ ❌❌❌")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")
    print("\n🤔 这是最后的调试步骤了。如果依然失败，请将最新的日志文件完整内容发给我。")
    print("   请运行: cat ~/.ai2thor/log/unity.log")
    print("="*50 + "\n")

finally:
    # 确保在程序结束时停止控制器
    if 'controller' in locals() and hasattr(controller, 'stop'):
        controller.stop()
        print("\n控制器已停止。")