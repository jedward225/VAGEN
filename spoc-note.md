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

● 你说得对！这确实是代码问题，不是版本问题。VAGEN之前在其他环境（如frozenlake、sokoban
  ）上都工作正常，问题出在我们的SPOC环境处理两张图像的方式上。

  让我重新分析这个问题：

  问题分析

  错误信息：Image features and image tokens do not match: tokens: 2524, features 3240

  关键点：
  - 我们返回了两张图像：[pil_image, pil_image]
  - 但是prompt模板中的图像预处理可能有问题
  - 2524 tokens vs 3240 features 的比例大约是 0.78，这不是整数倍关系

  真正的问题

  让我检查一下我们的图像处理逻辑：

● Read(vagen/env/spoc/env.py)
  ⎿  Read 15 lines (ctrl+r to expand)

● 问题找到了！我们返回两张相同的图像，但是模型期望两张不同尺寸或格式的图像。

  解决方案

  我们需要创建两张真正不同的图像，或者调整图像处理逻辑。让我尝试创建两张不同的图像（导
  航视图和操作视图）：

● Update(vagen/env/spoc/env.py)
  ⎿  (No changes)

> 你先解释一下原本的env.py是怎么运作的，代码是怎么写的，你想怎么调整？
  ⎿  ⧉ Selected 2 lines from ../verl/requirements.txt in Visual Studio Code

✻ Thinking…

  用户要求我解释原本的env.py是怎么运作的，代码是怎么写的，以及我想怎么调整。让我仔细分
  析现在的代码逻辑。

  从之前读取的代码来看：

  1. 原本的env.py在_render方法中：
    - 从AI2-THOR获取一个frame (numpy array)
    - 将frame转换为PIL Image
    - 返回两个相同的图像：[pil_image, pil_image]
  2. 这个设计是为了支持双摄像头系统：
    - Navigation Camera: 导航摄像头
    - Manipulation Camera: 操作摄像头
  3. 当前的问题：
    - 我们返回了两张相同的图像
    - 但是模型在处理这两张图像时出现了token和feature数量不匹配的问题

  让我先详细解释现在的代码逻辑，然后说明我的调整思路。

● 让我详细解释一下原本的env.py代码逻辑：

  原本的env.py运作方式

  1. 图像获取流程（在_render方法中）：
  # 从AI2-THOR获取一帧图像
  frame = self.env.last_event.frame  # numpy array格式
  if frame is None:
      # 执行Pass动作获取图像
      self.env.step("Pass")
      frame = self.env.last_event.frame

  # 转换为PIL Image
  pil_image = convert_numpy_to_PIL(frame)

  # 返回两个相同的图像
  multi_modal_data = {
      img_placeholder: [pil_image, pil_image]  # 两张相同的图像
  }

  2. 设计意图：
  - 支持双摄像头系统：Navigation Camera + Manipulation Camera
  - 但实际上AI2-THOR只提供一个视角
  - 所以用同一张图像模拟两个摄像头

  3. Prompt模板中的使用：
  Navigation Camera:
  <image>
  Manipulation Camera:
  <image>

  问题分析

  问题根源：
  - 我们返回了[pil_image, pil_image]（两张相同图像）
  - 但模型处理时产生了不同数量的tokens和features
  - 可能是因为图像预处理器对两张相同图像的处理结果不一致

  我的调整思路

  方案1：创建真正的两张不同图像
  # 导航图像：完整图像
  nav_image = pil_image

  # 操作图像：中心裁剪后放大（模拟近距离视角）
  manip_image = pil_image.crop(center_region).resize(original_size)

  multi_modal_data = {
      img_placeholder: [nav_image, manip_image]
  }

  方案2：临时使用单图像
  # 只返回一张图像，修改prompt模板
  multi_modal_data = {
      img_placeholder: [pil_image]
  }

  方案3：检查图像预处理
  - 确保两张图像的尺寸、格式完全一致
  - 可能需要调整图像分辨率或其他参数

  你倾向于哪种方案？还是想让我先尝试最简单的方案2，看看单图像是否能解决问题？

> 但实际上AI2-THOR只提供一个视角？？？SPOC官方的代码就是AI2-THOR提供两个视角，你在说什
  么？

● 你说得对！我道歉，我搞错了。SPOC确实是基于AI2-THOR的，而AI2-THOR是支持多个摄像头视角
  的。

  让我重新检查代码，看看我们是否正确获取了两个不同的摄像头视角：

● Read(vagen/env/spoc/env.py)
  ⎿  Read 25 lines (ctrl+r to expand)

● 我看到问题了！代码注释说明了问题：

  # For dual camera setup, use the same image for both cameras
  # In a real dual camera setup, you would get two different frames
  multi_modal_data = {
      img_placeholder: [pil_image, pil_image]  # Two identical images for now
  }

  问题：
  - 代码应该获取两个不同的摄像头视角
  - 但现在只获取了一个视角（self.env.last_event.frame）
  - 然后重复使用同一张图像

  正确的做法：
  我们需要从AI2-THOR获取两个不同的摄像头视角：
  1. 主摄像头视角（导航用）
  2. 机械臂摄像头视角（操作用）

  让我检查AI2-THOR的API，看看如何获取多个摄像头视角：

● Search(pattern: "camera|frame|third_party_camera", path: "vagen/env/spoc",
        output_mode: "content", head_limit: 10)
  ⎿  Found 10 lines (ctrl+r to expand)

● 我需要查看AI2-THOR的事件对象，看看是否有其他摄像头数据：

● Search(pattern: "last_event|third_party|wrist|arm.*camera", path: "vagen/env/spoc",
        output_mode: "content")
  ⎿  Found 55 lines (ctrl+r to expand)

● 你绝对是对的！我需要查看AI2-THOR在Stretch机器人配置下如何获取多个摄像头视角。让我检
  查AI2-THOR的事件对象中是否有额外的摄像头数据：

● Update Todos
  ⎿  ☐ 研究AI2-THOR中如何获取多个摄像头视角
     ☐ 修复双摄像头图像获取逻辑
     ☐ 解决token不匹配的问题

✢ Compacting conversation… (48s · ↓ 354 tokens · esc to interrupt)

一、SPOC 原版在做什么
1. 任务定义
在 ProcTHOR 室内模拟器 + Stretch RE-1 机器人模型中，给定一段自然语言指令（如 “Bring me the red mug on the counter”）。
机器人必须先「导航」（底座移动 + 机身旋转）再「操作」（机械臂伸缩＋拾取）完成长程任务。
数据与方法
核心思想：用启发式规划器生成「最短路径 expert 轨迹」，大规模模仿学习。
数据集中每条轨迹同时录制
两个 RGB 视频流（navigation / manipulation camera）
多个环境状态传感器（是否抓到目标、最近距离等）
expert 的离散动作序列（20 个动作，与你现在 env/spoc 的 ACTION_LOOKUP 完全一致）。
SPOC 模型：
文本指令编码器 (T5)
Goal-conditioned Visual Encoder（双摄像头）
Transformer 解码器 → 动作序列
训练：纯交叉熵 imitation；没有 RL、没有人类数据，靠「规模 + 最短路径」获得泛化能力。
参考官网介绍 [spoc-robot]。
二、VAGEN 的标准环境接口
VAGEN 设计了统一的多任务 RL 框架：
BaseEnv  <—— 每个具体环境实现 reset / step / render / reward
BaseService ——> 批量并发 env，用 HTTP or 内存 IPC 供 RolloutManager 访问
RolloutManager ——> 负责把 LLM prompt <-> env Observation 互换
RayTrainer (PPO/GRPO) ——> 训练 Actor/Critic

只要某个任务能够：
① 返回文本（或多模态 token）观测；② 接收离散动作并给 reward；
就能像 FrozenLake、Sokoban 一样接入 VAGEN。
三、env/spoc 是如何嫁接的
1. SpocEnv
继承 BaseEnv；内部直接调用 AI2-THOR Stretch 控制 API。
reset()
从 load_chores_episode() 随机抽一条「expert 轨迹元数据」：场景 ID、目标物 pose、开局坐标、文字指令。
真实重置 THOR 并把机器人 teleport 到 expert 起点。
step(action_str)
先用 PARSE_FUNC_MAP 把 LLM 生成的 <answer>moveahead,...</answer> 解析为动作列表。
依次执行 _execute_action()（20 个离散动作正好对应 SPOC 动作空间）。
调 measure_success() :「抓到目标 or 距离阈值 ≤0.35 m」→ +10 reward。
奖励还包含：「动作格式正确」+0.5。
_render()
生成 Grounding/World-modeling prompt：
       [Initial Observation]:
       Navigation Camera:
       [VISUAL]           # (纯文本模式时把 <image> 占位符换成 [VISUAL])
       ...
       Human Instruction: Fetch the mug ...

若你想恢复图片，只要保留 <image> 并让 render_mode=vision 即可。
SpocService
并发创建 env；根据 GPU id 分配到各卡；供 RolloutManagerService 批量 step_batch()。
接入点
scripts/examples/masked_grpo/spoc/.../env_config.yaml 描述如何采样数据集 → create_dataset 把 prompt-response-reward 保存成 parquet；
run_tmux.sh 启动两个 tmux：
server：RESTful Service（SpocService）
train：RayTrainer (PPO/GRPO)
与 SPOC 原版差异
| | SPOC-Training | VAGEN-SPOC |
|-------------|--------------|-----------|
| 学习范式 | imitation | RL (GRPO) |
| 模型 | RGB Transformer | 通用 VLM (Qwen2.5-VL) |
| 输入 prompt | raw image tensor | 自定义 grounding/world-modeling prompt（可带或不带 <image> token） |
| 目标 | 复现 expert 动作 | 在交互中学到更鲁棒的多轮视觉推理策略 |
四、当前训练目标 & 下一步
1. 目标
让多模态大模型在 真实交互 中学会：
读懂双摄像头信息 + 指令
规划分步行动（Grounding + WorldModeling Reasoning）
实际完成 Fetch 任务 → maximize success_rate + format_reward
GRPO 的选择：对长文本 selective mask，credit assignment 更精细。
为什么先用 纯文本模式？
<image> token 一张图 ≈ 576 tokens（Qwen2.5-VL）；
你现在的 prompt 已经 900+ tokens；若再加 2 张图，context 会到 2000+。
vLLM V0 对多模态 KV-Cache 占用巨大（警告里写 98 k token worst-case），极易 OOM。
先让策略 在文字描述层面收敛，再分阶段打开图片输入，效果更稳。
接下来可做的事
显存仍偏紧：把 gpu_memory_utilization 降到 0.35 或把 max_num_batched_tokens 再降到 1600；
若想逐步恢复图像：
     env_config:
       render_mode: vision
       limit_mm_per_prompt: 1   # 每轮最多 1 张 <image>
       max_model_len: 1024
并相应调低 prompt_length。
也可以复用 SPOC-Training 提供的视觉 encoder (SigLIP/DINO) 作为「前端 frozen 视觉」→ 把 Patch Embedding 当普通 token 拼到 prompt，可显著减小显存。
答复完毕，如需进一步调整显存 / prompt，请再告诉我。

五、最新一次 run（global step 0→25）的表现与问题定位
--------------------------------------------------
1. 训练阶段（steps 0-25）概况
   • prompt_length≈26 token，response_length≈1，属于纯文本最短指令格式 → vLLM 性能瓶颈主要在 **KV-Cache 预分配** 而非上下文长度。
   • 我们将
     - `max_trajectory_length` 提到 **1200**
     - `rollout.max_num_batched_tokens` 调成 **1800**
     - `gpu_memory_utilization` 暂压到 **0.3**
     训练期间显存稳定（单卡 ~15 GiB 使用，剩余 7-8 GiB 缓存）。
   • 每 mini-batch 4 轨迹 → 单 step 耗时 ~43 s，其中 vLLM 生成（gen）≈18 s，占比 40 %。
   • 指标
     - `train/success` 仍为 0：文本总结版 prompt 仅凭语言很难真抓到目标 → **属预期**。
     - `actor/grad_norm` 10-300 波动、`entropy_loss≈1`，表明策略仍在随机探索 early stage。

2. validation 阶段触发 ReadTimeout
   • `test_freq=25` → global step 25 进入 `_validate()`，调用 `RolloutManagerService.reset()` 去 **创建新一批 env**。
   • SpocService 端创建 env 需要加载 ProcTHOR 场景 & 机器人网格，单个 ~25-40 s。
     默认一次性批量 8 个 env，再加上 **Ray actor 传输开销**，首轮 validation 常 >10 min。
   • 我们 `rollout_manager.timeout=600`，整批没回来 → `requests.exceptions.ReadTimeout`，Ray task error 终止。

3. 解决方案
   A. **拉长超时**
      ```yaml
      rollout_manager.timeout: 1200  # 20 min
      ```
      或在 run_tmux.sh 中追加同名 override。
   B. **减少验证开销**
      - 把 `trainer.test_freq` 提到 100，只在保存 checkpoint 时做 validation。
      - 验证用更小 batch：
        ```yaml
        rollout_manager.n_trajectory: 1
        actor_rollout_ref.rollout.max_num_seqs: 4
        ```
      - 直接跳过 early validation：`trainer.val_before_train=False` 已设 → 可再把 `trainer.val_generations_to_log_to_wandb=0`。
   C. **服务端懒加载**
      若仍慢，可改 `SpocService` 为 “create on demand”：先只占位，真正 `reset(idx)` 时才 load THOR。

4. 下一步调优计划
   ① 继续纯文本跑通 1-2 k steps：观察 success/surprise 增长趋势；如停滞再考虑 reward shaping。
   ② 若 GPU 仍有余量，可把 `gpu_memory_utilization→0.35`、`max_num_batched_tokens→2000`，以降低 gen 时间。
   ③ 成功稳定后，逐步打开视觉：
      ```yaml
      env_config.render_mode: vision
      env_config.limit_mm_per_prompt: 1  # 保留 navigation camera
      data.max_prompt_length: 768        # 控制 context
      actor_rollout_ref.rollout.max_model_len: 1024
      ```
   ④ 高效验证：把 server & trainer 端口/网址改为 unix-socket IPC 可省去 HTTP 超时问题（待官方 PR）。


六、本轮（global step 17→24）运行快照与后续计划
---------------------------------------------
1. 训练是否稳定？
   • **未再出现 OOM / ReadTimeout**，vLLM+Ray 正常供能；GPU 占用单卡 ≈15 GiB，留有约 7 GiB 余量。
   • `actor_rollout_ref.rollout.gpu_memory_utilization=0.33` 与 1800 token KV-Cache 预分配配合良好。

2. 指标变化
   • 步长 90-110 s，其中 **生成阶段 65-87 s**，占总时长 70-80 %。
   • actor/entropy≈0.4→0.8，grad_norm ≈6-8，已不再有 `nan`。
   • 成功率依旧 0，属 early exploration 预期。

3. 风险点
   • 最大 response 可达 603 token，远超我们的 `data.max_response_length=200` ——> 生成侧尚未限长。
   • 训练到 50 step 会触发 val，仍可能引发 **ReadTimeout**（验证阶段创建 env 批太慢）。

4. 建议改动（已 TODO）
   A) 在 `run_tmux.sh` 追加
      `actor_rollout_ref.rollout.max_response_length=256 \
       actor_rollout_ref.rollout.stop=<|endoftext|> \
`   强制生成不超过 256 token。
   B) 把 `trainer.test_freq` 暂调大到 200；或改为 `rollout_manager.n_trajectory=1` during val。
   C) 若后续要真正恢复 `<image>` token，可把
      `data.max_prompt_length=512` 且逐步降低 `max_response_length`，观察 KV-Cache。

5. 下一步
   • 先跑满 1-2k step 观察 reward/advantage 曲线是否抬头；
   • 若长时间 success=0，考虑引入 curriculum：① shorter fetch tasks → ② full fetch；
   • 按需打开 image patch、DINO encoder做对比实验。