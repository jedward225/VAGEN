import ai2thor.controller

print("--- 开始测试AI2-THOR基础功能 ---")
controller = None
try:
    print("正在初始化控制器，请稍候...")
    # 使用最简单的默认配置启动
    controller = ai2thor.controller.Controller(
        scene="FloorPlan1",  # 使用最基础、最简单的场景
        gridSize=0.25,
        width=300,
        height=300,
    )
    print("✅ 控制器初始化成功！")

    print("正在重置场景...")
    event = controller.reset(scene="FloorPlan1")
    print("✅ 场景重置成功！")
    print(f"当前场景: {event.metadata['sceneName']}")

    print("尝试执行一个动作 'MoveAhead'...")
    event = controller.step(action="MoveAhead")
    print("✅ 动作执行成功！")

    print("\n🎉 AI2-THOR 核心功能看起来工作正常！")

except Exception as e:
    print(f"\n❌ 测试过程中发生错误: {e}")

finally:
    if controller:
        print("正在关闭控制器...")
        controller.stop()
        print("控制器已关闭。")