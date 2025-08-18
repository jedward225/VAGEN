# Embodied-Reasoner vs SPOC Prompt 系统详细对比分析

## 细节 1：系统提示的信息密度控制

### Embodied-Reasoner 的做法
```
You are a robot in given room. You need to complete the tasks according to human instructions. We provide an Available_Actions set and the corresponding explanations for each action. Each step, you should select one action from Available_Actions.
```

**特点分析**：
- **总长度**: 约200字符
- **信息量**: 仅包含角色定义、任务目标、行为约束三个核心要素
- **设计原理**: 认知负载最小化 - 避免给模型过多预设框架

### SPOC 当前做法
```
You are a Stretch robot performing object manipulation tasks in home environments.

VISUAL OBSERVATION:
You receive 3 separate visual inputs for comprehensive spatial awareness:
1. Navigation Camera: Wide-field view for spatial awareness and navigation
2. Manipulation Camera: Close-up view of objects within manipulation range  
3. Top-down Map: Bird's-eye view showing room layout, your path (blue), targets (red), current position (green)

ARM PROPRIOCEPTION:
Your arm state is always provided with these parameters:
- z: Arm extension (0.0m=retracted, positive=extended outward)
- y: Arm height (0.8m=default, can move up/down)  
- wrist: Wrist rotation angle in degrees
- gripper: Current gripper state (empty/holding_object_name)

AVAILABLE ACTIONS (20 total):
[详细的20个动作列表，包含技术参数]

COORDINATION STRATEGY:
[6点协调策略]

MANIPULATION WORKFLOW:
[9步操作流程]

[还有完整的示例对话]
```

**特点分析**：
- **总长度**: 8000+字符
- **信息量**: 包含环境描述、感知解释、动作详解、策略指导、工作流程、示例等
- **问题**: 信息过载可能分散模型注意力，预设过多可能限制灵活性

### 修改建议
保留核心信息，大幅精简系统提示：
```
You are a Stretch robot in a household environment. Your task is to find and fetch objects using navigation and manipulation actions. Select ONE action per step from the available actions.

Available Actions: moveahead, moveback, rotateright, rotateleft, pickup, dropoff, move_arm_out, move_arm_in, move_arm_up, move_arm_down
```

**修改原理**: 去除技术细节和预设策略，让模型基于观察自然推理

---

## 细节 2：思考过程的格式约束

### Embodied-Reasoner 的做法
```
Before making each decision, you can think, plan, and even reflect step by step, and then output your final action.
Your final action must strictly follow format: <DecisionMaking>Your Action</DecisionMaking>
```

**特点分析**：
- **思考自由度**: "can think, plan, and even reflect" - 完全开放的思考方式
- **格式要求**: 仅约束最终输出格式，不限制思考结构
- **设计哲学**: 相信模型的自然推理能力，不强加人为结构

### SPOC 当前做法
```
<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think>
<answer>action1, action2, action3</answer>
```

**特点分析**：
- **强制结构**: 必须按照observation→reasoning→prediction的固定流程
- **XML格式**: 严格的标签结构要求
- **问题**: 
  1. 不是所有推理都需要prediction步骤
  2. 固定结构可能不适合所有情况
  3. 模型可能为了满足格式而生成无意义内容

### 修改建议
采用自由思考 + 统一输出格式：
```
Think about the situation, then output your action.
Format: <DecisionMaking>action_name</DecisionMaking>

Example response:
I see a mug on the counter ahead. I need to get closer to pick it up.
<DecisionMaking>moveahead</DecisionMaking>
```

**修改原理**: 让模型按最自然的方式推理，避免格式化开销

---

## 细节 3：动作描述的详细程度

### Embodied-Reasoner 的做法
```
"navigate to <object>": Move to the object.
"pickup <object>": Pick up the object.
"put <object>": Put the item in your hand into or on the object.
"move forward": Move forward to see more clearly.
```

**特点分析**：
- **语义层面**: 描述动作的目的和效果，不涉及技术参数
- **简洁明了**: 每个动作一句话说明
- **用户友好**: 接近自然语言的描述方式

### SPOC 当前做法
```
- moveahead: Move the robot base forward by 0.2 meters
- rotateright: Rotate the robot base right by 30°
- move_arm_out: Extend the arm outward by 0.1 meters
- move_arm_up_small: Move the arm up by 0.02 meters (fine adjustment)
```

**特点分析**：
- **技术参数**: 包含精确的数值参数（0.2m, 30°, 0.1m等）
- **实现细节**: 描述了底层的执行机制
- **问题**: 
  1. 模型不需要知道具体的米数和角度
  2. 过多技术细节可能干扰语义理解
  3. 增加了prompt长度但没有提供有用信息

### 修改建议
采用语义化的动作描述：
```
Navigation: moveahead, moveback, rotateright, rotateleft
Manipulation: pickup, dropoff, move_arm_out, move_arm_in, move_arm_up, move_arm_down
```

**修改原理**: 模型只需要理解动作的语义含义，具体参数由环境处理

---

## 细节 4：上下文感知的动态提示

### Embodied-Reasoner 的做法
根据不同执行情况使用不同的后续提示：

**常规动作后**：
```
After executing your previous "{action}", you get this new image above.
To complete your task, you can think step by step at first and then output your new action.
```

**移动动作后**：
```
After executing your previous "{action}", you get this new image above.
You can use "navigate to <object>" to reach nearby, larger objects for closer inspection.
```

**错误情况**：
```
To complete your task, you can think step by step at first and then output your new action.
```

**特点分析**：
- **情境感知**: 根据上一步动作类型调整提示内容
- **针对性指导**: 移动后提醒可以导航到物体，提供相关建议
- **简洁性**: 即使在提供指导时也保持简洁

### SPOC 当前做法
```python
def action_template(**kwargs):
    # 所有情况都使用相同的模板
    return f"""After your answer, the extracted valid action is {valid_action}.
The environment feedback is: {env_feedback}
done: {done}
After that, the observation is:
{observation}
Arm State: {arm_state}
Human Instruction: {instruction}
Decide your next action(s)."""
```

**特点分析**：
- **静态模板**: 无论什么情况都使用相同的提示
- **信息堆砌**: 包含所有可能相关的信息
- **问题**: 
  1. 无法提供针对性指导
  2. 包含很多当前步骤不相关的信息
  3. 错过了根据执行结果优化提示的机会

### 修改建议
实现上下文感知的动态提示系统：

```python
def get_contextual_prompt(action, feedback, success):
    if "move" in action and success:
        return "After moving, check your new position and look for the target object."
    elif "pickup" in action:
        if success:
            return "Object picked up successfully. Navigate to the destination."
        else:
            return "Pickup failed. Adjust your position or arm and try again."
    elif "fail" in feedback.lower():
        return "Previous action failed. Consider an alternative approach."
    else:
        return "Continue with your task."
```

**修改原理**: 根据执行情况提供最相关的指导，减少无关信息

---

## 细节 5：示例对话的设计策略

### Embodied-Reasoner 的做法
```
Example:
Round 1:
Visual Observation: [Dual camera view]
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<think><observation>Navigation view: I am in a kitchen environment...</observation><reasoning>I need to navigate closer...</reasoning></think>
<answer>moveahead, moveahead</answer>
Round 2:
Env_feedback: Last action is executed successfully.
[继续简洁的对话示例]
```

**特点分析**：
- **最小示例**: 通常只有2-3轮对话
- **核心演示**: 重点展示思考→动作的基本流程
- **格式一致**: 严格按照要求的格式展示

### SPOC 当前做法
提供多个详细的示例对话，每个示例包含：
- 完整的4轮对话
- 详细的think内容（observation + reasoning + prediction）
- 复杂的多动作组合
- 不同格式的多个示例

**特点分析**：
- **示例冗长**: 每个示例都很详细完整
- **过度演示**: 可能给模型过多的模板化倾向
- **问题**: 
  1. 占用大量token空间
  2. 可能导致模型过度模仿示例格式
  3. 复杂示例可能干扰模型理解核心要求

### 修改建议
精简示例，突出核心模式：

```
Example:
Observation: I see a mug on the kitchen counter. My arm is retracted.
I need to get closer to the mug first.
<DecisionMaking>moveahead</DecisionMaking>

After moving closer:
Now I can see the mug clearly. I should extend my arm to reach it.
<DecisionMaking>move_arm_out</DecisionMaking>
```

**修改原理**: 简洁示例更容易理解核心模式，避免过度模仿

---

## 细节 6：多步动作的处理方式

### Embodied-Reasoner 的做法
```
Each step, you should select one action from Available_Actions.
Format: <DecisionMaking>Your Action</DecisionMaking>
```

**特点分析**：
- **单步原则**: 明确要求每次只选择一个动作
- **简化决策**: 避免复杂的多步规划
- **设计原理**: 单步决策更容易执行和调试，减少规划复杂度

### SPOC 当前做法
```
You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'.
IMPORTANT: Limit responses to 1-3 actions maximum.
<answer>action1, action2, action3</answer>
```

**特点分析**：
- **多步允许**: 允许一次执行1-3个动作
- **复杂规划**: 需要考虑动作间的顺序和依赖关系
- **问题**: 
  1. 增加了决策复杂度
  2. 动作顺序可能不合理
  3. 错误传播风险更高

### 修改建议
采用单步决策模式：

```
Select ONE action per step. This ensures better control and easier debugging.
Format: <DecisionMaking>single_action</DecisionMaking>
```

**修改原理**: 单步决策降低复杂度，提高成功率，便于错误定位

---

## 总结建议

基于以上6个细节的分析，建议的核心改进方向：

1. **极简化系统提示**: 从8000+字符减少到200字符左右
2. **自由化思考格式**: 取消强制XML结构，允许自然推理
3. **语义化动作描述**: 去除技术参数，保留语义含义
4. **动态化上下文提示**: 根据执行情况调整后续提示
5. **精简化示例**: 使用最小可理解的示例
6. **单步化决策**: 每次只执行一个动作

这些改进将显著减少认知负载，提高推理效率，增强系统的适应性和可维护性。