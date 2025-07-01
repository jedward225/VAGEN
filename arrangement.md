# SPOC × VAGEN 集成开发路线图

> 本文档用于跟踪**在 VAGEN 框架下新增 `spoc` 环境**的全部实施步骤与进度。请按照顺序推进，并在完成后勾选对应任务。

---

## 0. 项目目标
1. 在 `vagen/env` 下实现 `spoc` 环境（Stretch 机器人，20 离散动作）。  
2. 支持读取 **ChoresDataset**（`hdf5_sensors.hdf5` + `mp4`）并转成 VAGEN Episode。  
3. 使 `QwenVLRolloutManager` 能够 reset/step `spoc` 环境，并与 **GRPO / PPO** 训练脚本对接。  
4. 预留"上帝视角地图"接口，为未来研究做准备。

---

## 1. 目录骨架
```
vagen/env/spoc/
    ├─ env.py
    ├─ env_config.py
    ├─ prompt.py
    ├─ task_loader.py   # 新增
    ├─ service.py
    └─ service_config.py
```

---

## 2. 详细实施步骤

### 2.1 目录复制与基础替换
- [ ] **拷贝** `vagen/env/navigation` → `vagen/env/spoc`
- [ ] 全局替换模块引用：`navigation` → `spoc`
- [ ] 验证 `import SpocEnv` 不报错

### 2.2 `SpocEnvConfig` & `task_loader`
- [ ] 在 `env_config.py` 中新增 `SpocEnvConfig`（见示例字段）
- [ ] 创建 `task_loader.py` 并实现 `load_chores_episode()`
- [ ] `reset()` 能正常加载一条 Episode（本地单元测试）

### 2.3 20 动作执行逻辑
- [ ] 更新 `ACTION_LOOKUP` 至 20 个动作
- [ ] 实现 `_execute_action()` 对应 Stretch API
- [ ] 单元测试：随机动作序列跑 30 步无异常

### 2.4 Prompt & System Prompt
- [ ] `prompt.py` 中加入 Manipulation Actions 描述
- [ ] 增加 Fetch 任务 Few-shot 示例
- [ ] `env.system_prompt()` 返回结果通过 lint 检查

### 2.5 Service 层接入
- [ ] 新建 `SpocService`（复制 navigation/service.py 修改）
- [ ] 在 `verify_service.py` 中注册并跑通 reset→step→compute_reward

### 2.6 离线 IL（Masked-GRPO）
- [ ] 编写 `spoc_dataset_adapter.py` 供 `scripts/examples/masked_grpo/spoc/...` 使用
- [ ] 新建 YAML & `run_tmux.sh`，1 epoch 可训练

### 2.7 在线 RL 微调（Turn-PPO）
- [ ] 新建 `configs/spoc_rl_training.yaml`
- [ ] `scripts/examples/spoc_rl/run_tmux.sh` 跑通收集轨迹

### 2.8 指标与可视化
- [ ] 在 `env.step()` 的 info.metrics 添加 `success` / `pickup_success` 等
- [ ] WandB 曲线正确记录

### 2.9 God-View Map 接口（可选）
- [ ] `SpocEnv.get_navmesh()` 返回 occupancy grid `np.ndarray`
- [ ] Prompt 模板支持 `<map>` 占位符

### 2.10 文档与 CI
- [ ] 更新 `docs/envs/create-env.md` 添加 SPOC 示例
- [ ] 在 `scripts/benchmark` 下补充 spoc demo

---

## 3. 时间线（建议）
| 周次 | 目标 |
|-----|------|
| W1 | 完成 2.1 ~ 2.3 |
| W2 | 完成 2.4 ~ 2.5 |
| W3 | 完成 2.6；产出 `ckpt_imitation.pt` |
| W4 | 完成 2.7；RL 曲线收敛 |
| W5 | 2.8, 2.9, 2.10 & 清理代码 |

---

> **进度更新**：完成一项后请在对应 checkbox 打钩并简述 PR/commit。



一、在 vagen/env 下新增 SPOC 环境的落地步骤

VAGEN 的每个环境都由「5 + 1」个文件／模块组成：

```
vagen/env/<env_name>/
    ├─ env.py              # 继承 BaseEnv，定义 reset/step 等
    ├─ env_config.py       # @dataclass，继承 BaseEnvConfig
    ├─ prompt.py           # system_prompt 与模板
    ├─ service.py          # 继承 BaseService
    ├─ service_config.py   # @dataclass，继承 BaseServiceConfig
    └─ (可选) datasets/…   # 如需内置小规模 demo 数据
```

---

以下按照这一骨架分 6 步实现 SPOC 环境（env_name 暂定为 spoc）：

1. 目录复制

```bash
cp -r VAGEN/vagen/env/navigation VAGEN/vagen/env/spoc
```

复制后**逐文件**修改，避免从零敲代码同时保持接口一致。

2. env_config.py —— 配置项对齐

```python
@dataclass
class SpocEnvConfig(BaseEnvConfig):
    env_name: str = "spoc"
    resolution: int = 255
    prompt_format: str = "grounding_worldmodeling"
    # 额外：
    use_state_reward: bool = False        # 是否启用推理奖励
    success_threshold: float = 0.35       # 取物成功距离
    gpu_device: int = 0
    chores_split: str = "fifteen"         # chores 数据集子集
    task_type: str = "Fetch"              # 详见第二部分
```

- 其余字段（step_length、max_actions_per_step 等）可直接保留或按 Stretch 机器人步长微调。
- 记得更新 config_id() 中的字段列表。

3. env.py —— 核心逻辑

关键改动点示例（仅列出差异，细节自行查文档和 AI2-THOR API）：

```python
from ai2thor.platform import CloudRendering
from .task_loader import load_chores_episode   # 新写的 loader
# ---------- ACTIONS ----------
ACTION_LOOKUP = {
    # 导航 8 动作保持不变
    "moveahead": 1, "moveback": 2, "moveright": 3, "moveleft": 4,
    "rotateright": 5, "rotateleft": 6, "lookup": 7, "lookdown": 8,
    # Manipulation 动作
    "pickup": 9, "dropoff": 10,
    "move_arm_up": 11, "move_arm_down": 12, "move_arm_out": 13, "move_arm_in": 14,
    "wrist_open": 15, "wrist_close": 16,
    "move_arm_up_small": 17, "move_arm_down_small": 18,
    "move_arm_out_small": 19, "move_arm_in_small": 20,
}
# ---------- __init__ ----------
self.thor_config.update({
    "agentMode": "stretch",
    "renderInstanceSegmentation": True,
    "renderDepthImage": True,
})
# ---------- reset ----------
traj_data = load_chores_episode(self.config.chores_split,
                                self.config.task_type, idx)
# Teleport 起点、设置目标对象 id、语言指令等均由 traj_data 提供
# ---------- step ----------
# 1) 解析 LLM 返回动作字符串 -> tokens -> ACTION_LOOKUP idx
# 2) _execute_action：针对 20 动作分别调用 controller.step(...)
# 3) 计算 reward：
#    • 任务奖励：成功抓取并执行 "end" 给 +1，否则 0
#    • 动作罚分：和 navigation 相同 (-0.1)
# 4) done 条件：成功 / 超步数 / 失败撞墙过多
```

4. task_loader.py —— 读取 Chores 数据

```python
def load_chores_episode(split:str, task_type:str, idx:int):
    # 读取 hdf5_sensors.hdf5 & video 路径
    # 返回 dict = {
    #   'scene': 'FloorPlan11',
    #   'agentPose': {...},
    #   'targetObjectType': 'Mug',
    #   'targetObjectId': 'Mug|+00.1|+...|',
    #   'instruction': 'fetch a mug',
    #   'task_type': task_type,
    # }
```

- 仅加载元数据，不必解码视频帧（在线交互会重新渲染）。
- 若想做纯离线 IL，可另写 SpocOfflineDatasetAdapter 复用此 loader。

5. prompt.py —— 模板扩展

- 在 system_prompt() 中加入 Manipulation Actions 列表；
- example() 里给出完整「走到苹果→Pickup→走到 Microwave→Dropoff→End」五步示例；
- 其余保持 grounding_worldmodeling 协议，方便沿用 VAGEN Reasoning RL。

6. service.py／service_config.py —— 轻量改
- SpocService(BaseService) 里全部 NavigationEnv → SpocEnv；
- GPU 负载均衡代码直接复用；
- 在 step_batch() 外层保持 @service_state_reward_wrapper 不变即可让推理奖励透明接入。

至此，一个最小可运行的 spoc 环境就接入完毕，能被 QwenVLRolloutManager 正常 reset/step。

二、任务定义与 VAGEN 训练思路

1. 任务选择：Fetch（导航 + 拾取）
原因：
ObjectNav 仅 8 动作，很容易与现有 navigation 混淆；
Fetch 同时覆盖导航与操作，可完整验证 20 动作映射，又不涉及柜门、按钮等高复杂度交互（相比 Place、Open）。
成功判定：
机器人握持目标对象（an_object_is_in_hand==True 且 object_id==target_id）
与目标初始位置距离 ≤ 0.35 m
并在 1 s 内执行 end。
奖励：
成功 +1
每步 -0.01
无效动作 / 撞击 -0.05
Reasoning grounding/worldmodeling 奖励按 α,β 线性叠加（可在 yaml 控制）。

2. 训练路径（沿用 VAGEN 思路）

✔ 阶段 A：Masked-GRPO 离线模仿
输入：Chores Fetch 轨迹 (obs, expert_action)
env 设 render_mode: vision_offline → 直接使用预存图像帧或其视觉特征缓存，减小显存；
损失：交叉熵 + GRPO 蒸馏（如已有脚本 scripts/examples/masked_grpo/...）；
输出：ckpt_imitation.pt。

✔ 阶段 B：Turn-PPO + Reasoning Reward 微调
将 ckpt_imitation.pt 作为 pretrained_checkpoint 注入 configs/spoc_rl_training.yaml；
用在线 AI2-THOR 环境运行，多机 16 env 并行；
Advantage Estimator：BI_LEVEL_GAE（解决长程信用）；
α,β 初始 0，两万步内线性爬升到 0.3；

✔ 阶段 C（未来扩展）：God-View Map 强化
当需要利用“上帝视角地图”时，只需在 SpocEnv.get_env_state() 中额外返回 occupancy grid；
在 prompt 模板中插入 <map> 占位符，同时把 map ndarray 放入 multi_modal_data['<image>']，VAGEN 原生 pipeline 已能处理。

3. 与现有 navigation 的差异点

| 模块 | navigation | spoc_fetch |
|-----------------|-----------------------------|-----------------------------------|
| agentMode | default | stretch |
| 动作空间 | 8 | 20 |
| Camera | 单 RGB / multiview optional | 双 RGB（导航 + Manip） |
| 成功判定 | 距离 + done | pickup + 距离 + end |
| 奖励 | 任务 reward + 格式 reward | 任务 reward + 格式 + 推理 reward |
| 数据集 | 自带 json 脚本 | ChoresDataset hdf5 转 dict |

三、后续行动清单（开发者视角）

步骤	负责人	预计人天	交付/检查点
1. 创建 spoc 目录并批量替换 import	you	0.5	环境可 import SpocEnv
2. 实现 SpocEnvConfig & task_loader	you	1	reset() 能正确载入一条 episode
3. 完成 20-action _execute_action	you	1	单元测试：随机动作序列无报错
4. prompt 与 system_prompt 适配	you	0.5	env.system_prompt() 返回正确模板
5. Service 层打通	you	0.5	verify_service.py 通过
6. imitation 数据迭代器 & masked-grpo YAML	you	1	1 epoch 能跑通
7. RL YAML & tmux 脚本	you	0.5	RayPPOTrainer 开始收集 rollouts
8. wandb/metrics & success evaluator	you	0.5	曲线可视化
9. God-View 接口 stub	optional	0.5	get_navmesh() 返回 numpy grid
10. 文档 README 更新	you	0.5	docs/envs/create-env.md 添加 SPOC
完成前 1–7 步即可形成闭环；第 8 步后可开始超参调试；9、10 步为后续扩展。
