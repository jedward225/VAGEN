# FORMAT_CONFIGS for SPOC robot manipulation tasks
# Enhanced with dual-camera coordination and arm proprioception
FORMAT_CONFIGS = {
    "free_think": {
        "description": "You should first give your thought process, and then your answer.",
        "format": "<think>...</think><answer>...</answer>",
        "example": """<think>Navigation view: I can see the target mug on the kitchen counter ahead. Manipulation view: The counter surface is visible but the mug is not yet in my manipulation range. Arm state: retracted (z=0.0m, y=0.8m), wrist=0°, empty gripper. I need to navigate closer first, then extend my arm to pick it up.</think><answer>moveahead{action_sep}move_arm_out{action_sep}pickup</answer>

Alternative: <think>Navigation view: I see a book on a table to my right. Manipulation view: Table not visible yet. Arm state: retracted, empty. I need to turn right first, then approach and pick up.</think><answer>rotateright{action_sep}moveahead{action_sep}pickup</answer>"""
    },
    "no_think": {
        "description": "You should provide only your answer.",
        "format": "<answer>...</answer>",
        "example": """<answer>moveahead{action_sep}move_arm_out{action_sep}pickup</answer>

Alternative examples:
<answer>rotateright{action_sep}moveahead{action_sep}pickup</answer>
<answer>moveahead{action_sep}move_arm_down{action_sep}pickup</answer>"""
    },
    "grounding": {
        "description": "You should first give your thought process with your observation and reasoning, and finally your answer.\nThe observation should describe what you see in both navigation and manipulation views, and include your arm state.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "example": """<think><observation>Navigation view: I am in a kitchen. There is a counter to my left, a sink ahead, and a dining table to my right. The target mug appears to be on the dining table. Manipulation view: The table surface is visible but the mug is not yet in my manipulation range. Arm state: My arm is currently retracted (z=0.0m, y=0.8m), wrist angle is 0°, gripper is empty and not holding anything.</observation><reasoning>I need to move toward the dining table to get the mug within my manipulation range. I'll turn right and move forward to approach the table, then extend my arm to reach the mug.</reasoning></think><answer>rotateright{action_sep}moveahead{action_sep}move_arm_out</answer>

Alternative: <think><observation>Navigation view: I'm in a bedroom. I can see a phone on the nightstand directly ahead. Manipulation view: The nightstand edge is visible but phone is not in range. Arm state: Retracted (z=0.0m, y=0.8m), wrist 0°, gripper empty.</observation><reasoning>The phone is straight ahead on the nightstand. I just need to move closer and extend my arm to pick it up.</reasoning></think><answer>moveahead{action_sep}move_arm_out{action_sep}pickup</answer>"""
    },
}

# system_prompt function for SPOC robot manipulation tasks
def system_prompt(**kwargs):
    # This was previously returning prematurely, bug is now fixed.
    example = "" # Default empty example
    # Internally uses kwargs.get("format"), as in your original code
    selected_format = kwargs.get("format", "default")

    if selected_format in ["free_think", "default"]:
        example=f"""Example:
Round 1:
Visual Observation: [Navigation camera, Manipulation camera, Top-down map]
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<think>Navigation camera shows I'm in a kitchen with the target coffee mug on the counter ahead. Manipulation camera shows the counter edge but mug is not in range yet. The top-down map shows the target as a red circle on the counter ahead, and my current position as a green circle. My arm is retracted. I need to navigate closer to get the mug within manipulation range.</think>
<answer>moveahead, moveahead</answer>
Round 2:
Env_feedback: Last action is executed successfully.
Visual Observation: [Navigation camera, Manipulation camera, Top-down map] 
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<think>Good, I'm now closer to the counter. Navigation camera shows the mug is directly ahead. Manipulation camera shows the mug is now visible and within reach. The top-down map shows I've moved closer to the red target circle. I need to extend my arm out to reach the mug position.</think>
<answer>move_arm_out, move_arm_out, pickup</answer>
Round 3:
Env_feedback: Last action is executed successfully.
Visual Observation: [Navigation camera, Manipulation camera, Top-down map]
Arm State: z=0.2m, y=0.8m, wrist=0°, gripper=holding_mug
<think>Perfect! My arm is extended and I've successfully grasped the mug. Navigation camera shows I'm holding the target object. Manipulation camera confirms the mug is securely in my gripper. The top-down map shows my path leading to the target. Task completed successfully.</think>
<answer>Done - mug retrieved</answer>
Round 4:
Env_feedback: Success"""
    elif selected_format == "grounding":
        example=f"""Example:
Round 1:
Visual Observation: [Dual camera view]
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<think><observation>Navigation view: I am in a kitchen environment. There is a counter ahead with the target coffee mug visible on its surface. Manipulation view: I can see the counter edge but the mug is not yet within my manipulation range. Arm state: My arm is fully retracted (z=0.0m, y=0.8m), wrist is at neutral position (0°), and gripper is empty.</observation><reasoning>I need to navigate closer to the counter to bring the mug within my manipulation range. Moving forward will position me optimally for the pick-up task.</reasoning></think>
<answer>moveahead, moveahead</answer>
Round 2:
Env_feedback: Last action is executed successfully.
Visual Observation: [Dual camera view]
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<think><observation>Navigation view: I am now positioned closer to the counter with the mug clearly visible ahead. Manipulation view: The mug is now prominently displayed in my manipulation view and appears to be within reach. Arm state: My arm remains retracted (z=0.0m, y=0.8m), wrist at 0°, gripper ready for action.</observation><reasoning>The mug is now within my manipulation range. I need to extend my arm outward to reach the mug's position and then execute a pickup action.</reasoning></think>
<answer>move_arm_out, move_arm_out, pickup</answer>
Round 3:
Env_feedback: Last action is executed successfully.
Visual Observation: [Dual camera view]
Arm State: z=0.2m, y=0.8m, wrist=0°, gripper=holding_mug
<think><observation>Navigation view: I am still positioned at the counter, and I can see that I am now holding the target mug. Manipulation view: The mug is securely grasped in my gripper and clearly visible. Arm state: My arm is extended outward (z=0.2m, y=0.8m), wrist at neutral position, and gripper is successfully holding the target mug.</observation><reasoning>I have successfully retrieved the target object. The task has been completed as the mug is now in my possession.</reasoning></think>
<answer>Task completed - mug retrieved</answer>
Round 4:
Env_feedback: Success"""
    elif selected_format == "no_think":
        example=f"""Example:
Round 1:
Visual Observation: [Navigation camera, Manipulation camera, Top-down map]
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<answer>moveahead, moveahead</answer>
Round 2:
Env_feedback: Last action is executed successfully.
Visual Observation: [Navigation camera, Manipulation camera, Top-down map]
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<answer>move_arm_out, move_arm_out, pickup</answer>
Round 3:
Env_feedback: Last action is executed successfully.
Visual Observation: [Navigation camera, Manipulation camera, Top-down map]
Arm State: z=0.2m, y=0.8m, wrist=0°, gripper=holding_mug
<answer>Task completed</answer>
Round 4:
Env_feedback: Success"""
        
    base_prompt_text = """You are a Stretch robot performing object manipulation tasks in home environments.

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

Navigation Actions:
- moveahead: Move the robot base forward by 0.2 meters
- moveback: Move the robot base backward by 0.2 meters  
- rotateright: Rotate the robot base right by 30°
- rotateleft: Rotate the robot base left by 30°
- rotateright_small: Rotate the robot base right by 6°
- rotateleft_small: Rotate the robot base left by 6°
- lookup: [Deprecated for Stretch] - Do not use
- lookdown: [Deprecated for Stretch] - Do not use

Manipulation Actions:
- pickup: Initiate a grasp action to pick up an object in range
- dropoff: Execute a release action to drop the currently held object
- move_arm_up: Move the arm up by 0.1 meters
- move_arm_down: Move the arm down by 0.1 meters
- move_arm_out: Extend the arm outward by 0.1 meters
- move_arm_in: Retract the arm inward by 0.1 meters
- wrist_open: Rotate the wrist counterclockwise by 10°
- wrist_close: Rotate the wrist clockwise by 10°
- move_arm_up_small: Move the arm up by 0.02 meters (fine adjustment)
- move_arm_down_small: Move the arm down by 0.02 meters (fine adjustment)
- move_arm_out_small: Extend the arm outward by 0.02 meters (fine adjustment)
- move_arm_in_small: Retract the arm inward by 0.02 meters (fine adjustment)

MAP INTERPRETATION:
- Room layout: Walls, furniture, and navigable space
- Spatial relationships: Relative positions of rooms and objects
- Navigation planning: Use to understand the environment structure

COORDINATION STRATEGY:
1. Use the navigation camera to locate target objects and plan approach
2. Use the top-down map to understand room layout and spatial relationships
3. Use base movement actions to position yourself optimally
4. Use the manipulation camera to precisely guide arm movements
5. Monitor arm proprioception to ensure accurate positioning
6. Coordinate all three visual inputs to maintain spatial awareness during manipulation

MANIPULATION WORKFLOW:
1. Use the navigation camera to scan for target objects in the current view
2. Use the top-down map to understand room structure and plan navigation
3. Navigate to target using base movements while monitoring both cameras
4. Position yourself so target appears in manipulation camera view
5. Extend arm outward (move_arm_out) to reach target
6. Adjust arm height (move_arm_up/down) if needed
7. Fine-tune positioning with small movements if necessary
8. Execute pickup when object is within gripper range
9. Use dropoff when you need to release the object

The instruction will be provided with each observation. Use all three visual inputs (navigation camera, manipulation camera, top-down map) and arm proprioception to complete manipulation tasks efficiently."""
    return base_prompt_text + '\n' + example

# init_observation_template and action_template for SPOC robot manipulation
def init_observation_template(**kwargs):
    observation = kwargs.get("observation", "No observation provided.")
    instruction = kwargs.get("instruction", "No instruction provided.")
    arm_state = kwargs.get("arm_state", "z=0.0m, y=0.8m, wrist=0°, gripper=empty")
    
    return f"""[Initial Observation]:
{observation}
Arm State: {arm_state}
Human Instruction: {instruction}
Decide your next action(s)."""

def get_contextual_prompt(action, feedback, success, arm_state, observation, instruction):
    """根据执行情况生成上下文感知的提示"""
    
    # 处理action可能是list的情况
    if isinstance(action, list):
        action_str = action[0] if action else ""
    else:
        action_str = str(action) if action else ""
    
    # 检测可能的困境情况
    is_likely_stuck = (
        not success and 
        ("collision" in feedback.lower() or "fail" in feedback.lower()) and
        ("rotate" in action_str.lower() or "move" in action_str.lower())
    )
    
    # 基础信息
    base_info = f"""
Current observation: {observation}
Arm State: {arm_state}
Task: {instruction}"""
    
    # 根据动作类型和执行结果生成针对性提示
    if "pickup" in action_str.lower():
        if success:
            guidance = "Object picked up successfully. Navigate to the destination or complete the task."
        else:
            guidance = "Pickup failed. Adjust your position, extend your arm, or check if object is reachable."
    elif "move_arm" in action_str.lower() or "arm" in action_str.lower():
        if success:
            guidance = "Arm position adjusted. Check if target is now within reach for pickup."
        else:
            guidance = "Arm movement failed. Try alternative arm positioning or check for obstacles."
    elif "move" in action_str.lower() or "rotate" in action_str.lower():
        if success:
            guidance = "After moving, check your new position and look for the target object."
        else:
            if is_likely_stuck:
                guidance = "You appear to be stuck in a corner or tight space. Try 'moveback' to exit this area, then rotate to find an open path."
            elif "rotate" in action_str.lower():
                guidance = "Rotation failed or stuck in place. Try moving backward to exit tight spaces, then explore a different direction."
            else:
                guidance = "Movement failed. If in a corner or tight space, try moving backward first, then choose a clear direction."
    elif "fail" in feedback.lower() or not success:
        guidance = "Previous action failed. Consider an alternative approach to reach your goal."
    else:
        guidance = "Continue with your task. Select the next appropriate action."
    
    return f"""{guidance}
{base_info}
Decide your next action."""

def action_template(**kwargs):
    observation = kwargs.get("observation", "No observation provided.")
    instruction = kwargs.get("instruction", "No instruction provided.")
    valid_action = kwargs.get("valid_action", "No valid action provided.")
    env_feedback = kwargs.get("env_feedback", "No environment feedback provided.")
    arm_state = kwargs.get("arm_state", "z=0.0m, y=0.8m, wrist=0°, gripper=empty")
    
    # 判断执行是否成功
    success = "success" in env_feedback.lower() or "executed successfully" in env_feedback.lower()
    
    # 使用上下文感知的提示生成
    return get_contextual_prompt(
        action=valid_action,
        feedback=env_feedback, 
        success=success,
        arm_state=arm_state,
        observation=observation,
        instruction=instruction
    )

# format_prompt_generator function, similar to your first (FrozenLake) example
def format_prompt_generator(format_type):
    """
    Generates a prompt function for the specified robot navigation format type.
    This returned function creates the per-turn instruction for the LLM.
    """
    def prompt_function(**kwargs):
        """
        Generate a prompt for the specified format for the robot navigation task.
        
        Args:
            max_actions_per_step (int): Max actions. Defaults to 5 (common for robot).
            action_sep (str): Separator. Defaults to ',' (common for robot).
            add_example (bool): Whether to add an example. Defaults to True.
            
        Returns:
            str: The formatted prompt.
        """
        # Defaults suitable for the robot navigation task
        max_actions_per_step = kwargs.get("max_actions_per_step", 5)
        action_sep = kwargs.get("action_sep", ",")
        add_example = kwargs.get("add_example", True) # Default to True as per robot examples
        
        if format_type not in FORMAT_CONFIGS:
            raise ValueError(f"Unknown format_type: {format_type}")
        config = FORMAT_CONFIGS[format_type]
        
        base_prompt = f"""You can take up to {max_actions_per_step} action(s) at a time, separated by '{action_sep}'.
Limit responses to 1-3 actions maximum. 
{config["description"]}"""
        
        if "additional_info" in config: # In case it's added to FORMAT_CONFIGS later
            base_prompt += f"\n{config['additional_info']}"
        
        base_prompt += f"""
Your response should be in the format of:
{config["format"]}"""
        
        if add_example:
            # The 'e.g.' is already part of the example string in this FORMAT_CONFIGS
            example_text = config["example"].format(action_sep=action_sep)
            return base_prompt + '\n' + f"e.g. {example_text}"
        
        return base_prompt
    
    return prompt_function

# format_prompt dictionary, as in your first (FrozenLake) example
format_prompt = {
    ft: format_prompt_generator(ft) 
    for ft in FORMAT_CONFIGS  # Iterate directly over keys in FORMAT_CONFIGS
}


def get_action_explanations():
    """返回详细的动作解释，参考embodied-reasoner风格"""
    return """
Available Actions:
Navigation:
- "moveahead": Move the robot base forward by 0.2 meters to approach objects or explore.
- "moveback": Move the robot base backward by 0.2 meters to create distance or reposition.
- "rotateright": Rotate the robot base right by 30° to change viewing direction or orientation.
- "rotateleft": Rotate the robot base left by 30° to change viewing direction or orientation.
- "rotateright_small": Rotate the robot base right by 6° for fine directional adjustments.
- "rotateleft_small": Rotate the robot base left by 6° for fine directional adjustments.

Manipulation:
- "pickup": Initiate a grasp action to pick up an object currently in manipulation range.
- "dropoff": Execute a release action to drop the currently held object.
- "move_arm_up": Move the arm up by 0.1 meters to reach higher objects or avoid obstacles.
- "move_arm_down": Move the arm down by 0.1 meters to reach lower objects or surfaces.
- "move_arm_out": Extend the arm outward by 0.1 meters to reach distant objects.
- "move_arm_in": Retract the arm inward by 0.1 meters to bring objects closer or reset position.
- "wrist_open": Rotate the wrist counterclockwise by 10° to adjust gripper orientation.
- "wrist_close": Rotate the wrist clockwise by 10° to adjust gripper orientation.
- "move_arm_up_small": Move the arm up by 0.02 meters for precise height adjustments.
- "move_arm_down_small": Move the arm down by 0.02 meters for precise height adjustments.
- "move_arm_out_small": Extend the arm outward by 0.02 meters for precise reach adjustments.
- "move_arm_in_small": Retract the arm inward by 0.02 meters for precise positioning."""

if __name__ == "__main__":
    # Example usage
    max_actions_per_step = 2
    action_sep = "|"
    
    for key, func in format_prompt.items():
        print(f"{key} format prompt:")
        print(func(max_actions_per_step=max_actions_per_step, action_sep=action_sep))
        print("\n" + "="*50 + "\n")