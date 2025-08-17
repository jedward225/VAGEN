# SPOC Environment Analysis

## Installation Commands
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
sudo apt-get update -y && sudo apt-get install -y vulkan-tools mesa-vulkan-drivers

pip install ai2thor
sudo apt-get update
sudo apt-get install -y vulkan-tools libvulkan1 vulkan-utils
sudo apt-get install -y xvfb mesa-utils
sudo apt install net-tools
sudo apt-get install -y libvulkan1
sudo apt install x11-apps
```

---

# 🤖 Agent Prompt 完整分析报告

## 概述
SPOC环境给GPT-4o-mini agent提供的是一个多层次、结构化的prompt系统，包含系统提示、历史反馈、当前观察和格式要求。

---

## 📋 Prompt 结构组成

### 1. **System Prompt** (总体指导)

#### 1.1 任务定义部分 (已优化)
**当前使用的System Prompt:**
```
You are a Stretch robot in a household environment. Your task is to find and fetch specific objects.
STRATEGY: 
1) Clarify your tasks and goals, use the map properly to explore the room, and navigate to the appropriate location
2) Try to find the target, extend your arm and pickup the object

IMPORTANT: Keep responses concise. 
In <think> tags: observation, reasoning and prediction
In <answer> tags: action name(s) only

VALID ACTIONS: moveahead, moveback, rotateright, rotateleft, rotateright_small, rotateleft_small, pickup, dropoff, move_arm_up, move_arm_down, move_arm_out, move_arm_in, wrist_open, wrist_close, move_arm_up_small, move_arm_down_small, move_arm_out_small, move_arm_in_small
You cannot use any other actions not listed above.
```

**优化要点:**
- ✅ **强调地图使用**: "use the map properly to explore" - 提高空间感知
- ✅ **简化策略**: 从3步减为2步，更直接高效
- ✅ **明确动作列表**: 直接列出所有可用动作
- ✅ **减少冗余**: 移除过于详细的思考格式要求

#### 1.2 动作约束
- **有效动作列表**(20个): moveahead, moveback, rotateright, rotateleft, pickup, dropoff, move_arm_up, move_arm_down, move_arm_out, move_arm_in, wrist_open, wrist_close + 小幅度调整版本
- **禁用动作**: moveleft, moveright, lookup, lookdown

#### 1.3 输入输出格式 (`free_think` - 已优化)
**当前使用格式:**
```
<think>...</think>
<answer>...</answer>
```

**与原版对比:**
- ❌ **原版** `grounding_worldmodeling`: 需要详细的 observation/reasoning/prediction 三段式
- ✅ **优化版** `free_think`: 简化为一个 think 标签，更灵活高效
- 🎯 **效果**: 减少token消耗，提高响应速度，保持核心思考能力

#### 1.4 环境理解指导
- **机械臂状态**: z(伸展)/y(高度)/wrist(腕部角度)/gripper(夹爪状态)
- **三视图协调**: 导航相机、操作相机、俯视地图
- **8步操作流程**: 扫描→规划→导航→定位→伸展→调整→抓取→释放

#### 1.5 奖励机制
- 格式正确: +0.5
- 成功操作: +10.0

### 2. **Per-Turn Prompt** (每轮动态信息)

#### 2.1 历史动作反馈模板 (action_template)
```
After your answer, the extracted valid action is [上次提取的动作].
The environment feedback is: [成功/失败信息]
reward: [奖励值]
done: [任务完成状态]
After that, the observation is:
[当前观察...]
Arm State: [机械臂状态]
Human Instruction: [任务指令]
Decide your next action(s).
```

#### 2.2 环境反馈类型
- **成功**: `"Action 'moveahead' executed successfully."`
- **失败**: `"Action 'pickup' FAILED - No object in range. Try a different approach."`
- **无效**: `"No valid action was provided. Please specify a valid action."`

#### 2.3 当前观察信息
- **视觉描述**: 三个摄像头的文字描述
- **图像数据**: 3个图像占位符 `<image> <image> <image>`
  - 导航相机图像 (384x224)
  - 操作相机图像 (384x224) 
  - 俯视地图图像 (396x224，带轨迹)
- **机械臂状态**: `z=0.1m, y=0.8m, wrist=15°, gripper=holding_mug`
- **任务指令**: `"Find and fetch the coffee mug"`

### 3. **Format Prompt** (行为约束 - 已优化)
**当前设置:**
```
You can take up to 3 action(s) at a time, separated by ','.
IMPORTANT: Limit responses to 1-3 actions maximum. Avoid repeating actions.
You should first give your thought process with your observation, reasoning, and prediction of next state, then your answer.
Include both camera views and arm proprioception in observation and prediction.

Your response should be in the format of:
<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>
```

---

## 📊 完整的Multi-Turn对话示例

### Round 1 (初始)
**User Prompt:**
```
[Initial Observation]:
Visual Observation: [Navigation camera showing kitchen, Manipulation camera showing counter edge, Top-down map showing robot position] <image> <image> <image>
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
Human Instruction: Find and fetch the coffee mug
Decide your next action(s).
```

**Agent Response:**
```
<think>
<observation>Navigation view: I am in a kitchen environment with a counter directly ahead. The target coffee mug is visible on the counter surface. Manipulation view: I can see the counter edge but the mug is not yet within reach. Arm state: Retracted (z=0.0m, y=0.8m), wrist neutral, gripper empty.</observation>
<reasoning>I need to navigate closer to the counter to bring the mug within manipulation range.</reasoning>
<prediction>Navigation view: I will be closer to the counter with the mug accessible. Manipulation view: The mug will be visible and within arm's reach. Arm state: Still retracted but in optimal position.</prediction>
</think>
<answer>moveahead, moveahead</answer>
```

### Round 2 (带历史反馈)
**User Prompt:**
```
After your answer, the extracted valid action is moveahead, moveahead.
The environment feedback is: Action 'moveahead' executed successfully.
reward: 0.5
done: False
After that, the observation is:
Visual Observation: [Navigation camera showing closer view of mug, Manipulation camera showing mug in view, Top-down map showing updated position] <image> <image> <image>
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
Human Instruction: Find and fetch the coffee mug
Decide your next action(s).
```

---

## 🔍 关键特性分析

### 1. **历史感知能力**
- Agent能获得**上一轮动作的执行结果**
- 包含**成功/失败的具体原因**
- 提供**奖励反馈**引导学习

### 2. **多模态输入**
- **文字描述** + **3个实时图像**
- **机械臂本体感知**数据
- **空间地图信息**（带轨迹显示）

### 3. **格式强约束**
- 严格要求特定的XML标签格式
- 限制动作数量（1-3个）
- 禁止使用未列出的动作

### 4. **逐步引导策略**
- 观察 → 推理 → 预测 → 行动
- 鼓励**多步规划**和**状态预测**
- 强调**视觉-运动协调**

### 5. **错误纠正机制**
- 失败动作会提供**具体失败原因**
- 建议"尝试不同方法"
- 通过奖励信号强化正确行为

---

## 📈 Prompt设计优势

1. **结构化思维**: 强制agent进行观察-推理-预测的完整思考流程
2. **多模态融合**: 有效整合视觉、本体感知和地图信息
3. **在线学习**: 通过即时反馈不断调整策略
4. **错误容忍**: 提供失败信息而非简单惩罚
5. **任务导向**: 明确的分步策略指导

这个prompt系统为robot learning提供了一个非常全面和结构化的交互框架。

---

## 🧪 测试配置优化

### GPT-4o测试配置 (`gpt4o_spoc.py`)

#### **环境配置**
```python
env_config = SpocEnvConfig(
    prompt_format="free_think",      # 使用简化格式
    max_actions_per_step=3,          # 保持灵活性
    action_sep=",",                  # 逗号分隔动作
    image_placeholder="<image>"      # 三图像输入
)
```

#### **模型配置**  
```python
MODEL_CONFIG = {
    "model_name": "gpt-4o-mini",
    "max_tokens": 1024,              # 🎯 已优化: 增加输出空间
    "temperature": 0.2,              # 低随机性，提高一致性
    "seed": 42                       # 固定种子便于复现
}
```

#### **测试设置**
```python
TEST_CONFIG = {
    "task_types": ["ObjectNavType", "RoomNav", "RoomVisit"],  # 🎯 已优化: 多任务类型
    "episodes_per_task": 2,          # 🎯 已优化: 每类2个episode
    "max_steps": 60,                 # 充分的探索时间
    "save_images": True,             # 保存详细可视化
    "image_interval": 5,             # 🎯 已优化: 每5步保存一次
    "debug_mode": True               # 详细日志输出
}
```

---

## 🗺️ 地图生成问题解决方案

### 问题背景
- 原始代码使用复杂的`SPOCMapVisualizer`存在兼容性问题
- `VisualizeWaypoints`在当前AI2-THOR版本中不支持
- 需要可靠的top-down地图生成带轨迹显示

### 解决方案
使用环境内置的`env.get_top_down_map(include_path=True, path_width=0.15)`方法：

```python
# 正确的地图生成方法
enhanced_map = self.env.get_top_down_map(include_path=True, path_width=0.15)
map_img = Image.fromarray(enhanced_map).convert('RGB')
```

### 关键修改
1. 删除`gpt4o_mini_spoc_vis.py`依赖
2. 简化`_create_combined_image()`方法
3. 使用经过验证的地图生成API
4. 添加JSON序列化修复用于NumPy类型

### 结果
- ✅ 生成带橙色轨迹线的俯视地图
- ✅ 路径宽度0.15，清晰可见
- ✅ 兼容当前AI2-THOR版本
- ✅ 无需复杂的坐标转换
