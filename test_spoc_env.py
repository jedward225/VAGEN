#!/usr/bin/env python3

import sys
import os
sys.path.append('/root/VAGEN')

# 测试SPOC环境的直接返回格式
def test_spoc_env():
    try:
        from vagen.env.spoc.env_config import SpocEnvConfig
        from vagen.env.spoc.env import SpocEnv
        
        print("=== 测试SPOC环境直接调用 ===")
        
        # 创建配置
        config = SpocEnvConfig()
        print(f"配置创建成功: {config}")
        
        # 创建环境
        print("正在创建SPOC环境...")
        env = SpocEnv(config)
        print("环境创建成功!")
        
        # 重置环境
        print("正在重置环境...")
        obs, info = env.reset(seed=42)
        
        print("=== 环境重置成功 ===")
        print(f"观察类型: {type(obs)}")
        print(f"观察键: {obs.keys() if isinstance(obs, dict) else 'Not a dict!'}")
        
        if isinstance(obs, dict):
            print("\n=== 观察内容详情 ===")
            for key, value in obs.items():
                print(f"  {key}: {type(value)} - {str(value)[:100]}...")
        
        print(f"\n信息类型: {type(info)}")
        print(f"信息内容: {info}")
        
        # 测试_render方法直接调用
        print("\n=== 测试_render方法直接调用 ===")
        render_result = env._render(init_obs=True)
        print(f"_render返回类型: {type(render_result)}")
        print(f"_render返回键: {render_result.keys() if isinstance(render_result, dict) else 'Not a dict!'}")
        
        if isinstance(render_result, dict):
            for key, value in render_result.items():
                print(f"  {key}: {type(value)} - {str(value)[:100]}...")
        
        # 关闭环境
        env.close()
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_spoc_env() 