#!/usr/bin/env python3
"""
简单测试俯视图功能
"""

import os
import sys
sys.path.append('/home/jiajunliu/VAGEN')

from vagen.env.spoc.env_config import SpocEnvConfig
from vagen.env.spoc.env import SpocEnv

SPOC_DATA_PATH = os.environ.get("SPOC_DATA_PATH", "/home/jiajunliu/spoc_data/fifteen") 
os.environ["SPOC_DATA_PATH"] = SPOC_DATA_PATH

config = SpocEnvConfig(
    data_path=SPOC_DATA_PATH,
    task_type="FetchType", 
    chores_split="train"
)

env = SpocEnv(config)

# 重置环境
obs, info = env.reset(seed=12345)
scene = env.episode_data.get('scene', 'unknown')
print(f"场景: {scene}")

# 保存第一视角图片
if '<image>' in obs.get('multi_modal_data', {}):
    obs['multi_modal_data']['<image>'][0].save("first_person_view.png")
    print("保存了第一视角图片")

# 生成俯视图
obs, reward, done, info = env.step("get_map")

# 保存俯视图
map_placeholder = getattr(config, "map_placeholder", "<map>")
if map_placeholder in obs.get('multi_modal_data', {}):
    obs['multi_modal_data'][map_placeholder][0].save("topdown_map.png")
    print("成功！俯视图已保存")
    print("对比 first_person_view.png 和 topdown_map.png")
else:
    print("失败：没有生成俯视图")

env.close()