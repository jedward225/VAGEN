# FORMAT_CONFIGS for SPOC robot manipulation tasks
# Enhanced with dual-camera coordination and arm proprioception
FORMAT_CONFIGS = {
    "free_think": {
        "description": "You should first give your thought process, and then your answer.",
        "format": "<think>...</think><answer>...</answer>",
        "example": """<think>Navigation camera: I can see the target mug on the kitchen counter ahead. Manipulation camera: The counter surface is visible but the mug is not yet in my manipulation range. Arm state: retracted (z=0.0m, y=0.8m), wrist=0°, empty gripper. I need to navigate closer first, then extend my arm to pick it up.</think><answer>moveahead{action_sep}moveahead{action_sep}move_arm_out{action_sep}pickup</answer>"""
    },
    "no_think": {
        "description": "You should provide only your answer.",
        "format": "<answer>...</answer>",
        "example": """<answer>moveahead{action_sep}moveahead{action_sep}move_arm_out{action_sep}pickup</answer>"""
    },
    "grounding": {
        "description": "You should first give your thought process with your observation and reasoning, and finally your answer.\nThe observation should describe what you see in both navigation and manipulation cameras, and include your arm state.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning></think><answer>...</answer>",
        "example": """<think><observation>Navigation camera: I am in a kitchen. There is a counter to my left, a sink ahead, and a dining table to my right. The target mug appears to be on the dining table. Manipulation camera: The table surface is visible but the mug is not yet in my manipulation range. Arm state: My arm is currently retracted (z=0.0m, y=0.8m), wrist angle is 0°, gripper is empty and not holding anything.</observation><reasoning>I need to move toward the dining table to get the mug within my manipulation range. I'll turn right and move forward to approach the table, then extend my arm to reach the mug.</reasoning></think><answer>rotateright{action_sep}moveahead{action_sep}moveahead{action_sep}move_arm_out</answer>"""
    },
    "worldmodeling": {
        "description": "You should first give your thought process with reasoning and prediction of next state, then your answer.\nThe prediction should describe what you expect to see in both cameras and your arm state after actions.",
        "format": "<think><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": """<think><reasoning>I can see the target mug on the table through my navigation camera. I need to get closer to it and position my arm properly for grasping. My manipulation camera will show the mug clearly once I'm in range.</reasoning><prediction>Navigation camera: I will be standing next to the dining table with the mug clearly visible. Manipulation camera: The mug will be prominently displayed in the center of my view, within grasping distance. Arm state: My arm will be extended outward (z=0.2m, y=0.8m), wrist positioned for optimal grasping, gripper ready to close around the mug.</prediction></think><answer>rotateright{action_sep}moveahead{action_sep}moveahead{action_sep}move_arm_out{action_sep}move_arm_out</answer>"""
    },
    "grounding_worldmodeling": {
        "description": "You should first give your thought process with your observation, reasoning, and prediction of next state, then your answer.\nInclude both camera views and arm proprioception in observation and prediction.",
        "format": "<think><observation>...</observation><reasoning>...</reasoning><prediction>...</prediction></think><answer>...</answer>",
        "example": """<think><observation>Navigation camera: I am in a kitchen with a dining table to my right. The target mug is visible on the table surface. Manipulation camera: The table edge is visible but the mug is not yet in my manipulation range. Arm state: My arm is retracted (z=0.0m, y=0.8m), wrist at 0°, gripper empty and ready.</observation><reasoning>I need to move closer to the table to get the mug within my manipulation range. I'll turn right and move forward to approach the table, then extend my arm outward to reach the mug.</reasoning><prediction>Navigation camera: I will be standing close to the dining table with the mug clearly visible. Manipulation camera: The mug will be prominently displayed and within grasping distance. Arm state: My arm will be extended outward (z=0.2m, y=0.8m), positioned optimally for grasping the mug.</prediction></think><answer>rotateright{action_sep}moveahead{action_sep}moveahead{action_sep}move_arm_out{action_sep}move_arm_out</answer>"""
    }
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
Navigation Camera:
<nav_image>
Manipulation Camera:
<manip_image>
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<think>Navigation camera shows I'm in a kitchen with the target coffee mug on the counter ahead. Manipulation camera shows the counter edge but mug is not in range yet. My arm is retracted. I need to navigate closer to get the mug within manipulation range.</think>
<answer>moveahead, moveahead</answer>
Round 2:
Env_feedback: Last action is executed successfully.
Navigation Camera:
<nav_image>
Manipulation Camera:
<manip_image>
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<think>Good, I'm now closer to the counter. The navigation camera shows the mug is directly ahead. Manipulation camera shows the mug is now visible and within reach. I need to extend my arm out to reach the mug position.</think>
<answer>move_arm_out, move_arm_out, pickup</answer>
Round 3:
Env_feedback: Last action is executed successfully.
Navigation Camera:
<nav_image>
Manipulation Camera:
<manip_image>
Arm State: z=0.2m, y=0.8m, wrist=0°, gripper=holding_mug
<think>Perfect! My arm is extended and I've successfully grasped the mug. Navigation camera shows I'm holding the target object. Manipulation camera confirms the mug is securely in my gripper. Task completed successfully.</think>
<answer>Done - mug retrieved</answer>
Round 4:
Env_feedback: Success"""
    elif selected_format == "grounding":
        example=f"""Example:
Round 1:
Navigation Camera:
<nav_image>
Manipulation Camera:
<manip_image>
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<think><observation>Navigation camera: I am in a kitchen environment. There is a counter ahead with the target coffee mug visible on its surface. Manipulation camera: I can see the counter edge but the mug is not yet within my manipulation range. Arm state: My arm is fully retracted (z=0.0m, y=0.8m), wrist is at neutral position (0°), and gripper is empty.</observation><reasoning>I need to navigate closer to the counter to bring the mug within my manipulation range. Moving forward will position me optimally for the pick-up task.</reasoning></think>
<answer>moveahead, moveahead</answer>
Round 2:
Env_feedback: Last action is executed successfully.
Navigation Camera:
<nav_image>
Manipulation Camera:
<manip_image>
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<think><observation>Navigation camera: I am now positioned closer to the counter with the mug clearly visible ahead. Manipulation camera: The mug is now prominently displayed in my manipulation view and appears to be within reach. Arm state: My arm remains retracted (z=0.0m, y=0.8m), wrist at 0°, gripper ready for action.</observation><reasoning>The mug is now within my manipulation range. I need to extend my arm outward to reach the mug's position and then execute a pickup action.</reasoning></think>
<answer>move_arm_out, move_arm_out, pickup</answer>
Round 3:
Env_feedback: Last action is executed successfully.
Navigation Camera:
<nav_image>
Manipulation Camera:
<manip_image>
Arm State: z=0.2m, y=0.8m, wrist=0°, gripper=holding_mug
<think><observation>Navigation camera: I am still positioned at the counter, and I can see that I am now holding the target mug. Manipulation camera: The mug is securely grasped in my gripper and clearly visible. Arm state: My arm is extended outward (z=0.2m, y=0.8m), wrist at neutral position, and gripper is successfully holding the target mug.</observation><reasoning>I have successfully retrieved the target object. The task has been completed as the mug is now in my possession.</reasoning></think>
<answer>Task completed - mug retrieved</answer>
Round 4:
Env_feedback: Success"""
    elif selected_format == "worldmodeling":
        example=f"""Example:
Round 1:
Navigation Camera:
<nav_image>
Manipulation Camera:
<manip_image>
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<think><reasoning>I can see the target coffee mug on the counter through my navigation camera. It's positioned ahead of me but not yet within my manipulation range. My arm is currently retracted and I need to get closer to the counter for successful manipulation.</reasoning><prediction>Navigation camera: I will be standing closer to the counter with the mug clearly visible and accessible. Manipulation camera: The mug will be prominently displayed in my manipulation view, positioned optimally for grasping. Arm state: My arm will remain retracted (z=0.0m, y=0.8m) but I'll be in position to extend it toward the mug.</prediction></think>
<answer>moveahead, moveahead</answer>
Round 2:
Env_feedback: Last action is executed successfully.
Navigation Camera:
<nav_image>
Manipulation Camera:
<manip_image>
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<think><reasoning>Perfect positioning achieved. The mug is now clearly visible in both cameras and within my manipulation range. I need to extend my arm outward to reach the mug's position and then execute a pickup action to grasp it.</reasoning><prediction>Navigation camera: I will maintain my position at the counter while holding the mug. Manipulation camera: The mug will be securely grasped and visible in my gripper. Arm state: My arm will be extended outward (z=0.2m, y=0.8m) with the mug firmly held in my gripper.</prediction></think>
<answer>move_arm_out, move_arm_out, pickup</answer>
Round 3:
Env_feedback: Last action is executed successfully.
Navigation Camera:
<nav_image>
Manipulation Camera:
<manip_image>
Arm State: z=0.2m, y=0.8m, wrist=0°, gripper=holding_mug
<think><reasoning>Excellent! I have successfully grasped the target mug. My arm is properly extended and the gripper has secured the object. The task objective has been achieved.</reasoning><prediction>Task completed successfully. I will maintain my current position with the mug securely held, ready for any subsequent instructions or task completion confirmation.</prediction></think>
<answer>Task completed - mug retrieved</answer>
Round 4:
Env_feedback: Success"""
    elif selected_format == "grounding_worldmodeling":
        example=f"""Example:
Round 1:
Navigation Camera:
<nav_image>
Manipulation Camera:
<manip_image>
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<think><observation>Navigation camera: I am in a kitchen environment with a counter directly ahead. The target coffee mug is visible on the counter surface. Manipulation camera: I can see the counter edge in my manipulation view, but the mug is not yet within my manipulation range. Arm state: My arm is fully retracted (z=0.0m, y=0.8m), wrist is at neutral position (0°), and gripper is empty and ready.</observation><reasoning>I need to navigate closer to the counter to bring the mug within my manipulation range. Moving forward will position me optimally for the subsequent manipulation actions.</reasoning><prediction>Navigation camera: I will be standing closer to the counter with the mug clearly visible and accessible. Manipulation camera: The mug will be prominently displayed in my manipulation view, positioned optimally for arm extension and grasping. Arm state: My arm will remain retracted but I'll be in optimal position to extend toward the mug.</prediction></think>
<answer>moveahead, moveahead</answer>
Round 2:
Env_feedback: Last action is executed successfully.
Navigation Camera:
<nav_image>
Manipulation Camera:
<manip_image>
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<think><observation>Navigation camera: I am now positioned optimally at the counter with the mug clearly visible ahead. Manipulation camera: The mug is now prominently displayed in my manipulation view and appears to be within reach of my arm. Arm state: My arm remains retracted (z=0.0m, y=0.8m), wrist at neutral position (0°), gripper empty and ready for pickup action.</observation><reasoning>The mug is now within my manipulation range. I need to extend my arm outward to reach the mug's position and then execute a pickup action to secure the object.</reasoning><prediction>Navigation camera: I will maintain my position at the counter while successfully holding the target mug. Manipulation camera: The mug will be securely grasped and clearly visible in my gripper. Arm state: My arm will be extended outward (z=0.2m, y=0.8m) with the mug firmly held in my gripper, task completed.</prediction></think>
<answer>move_arm_out, move_arm_out, pickup</answer>
Round 3:
Env_feedback: Last action is executed successfully.
Navigation Camera:
<nav_image>
Manipulation Camera:
<manip_image>
Arm State: z=0.2m, y=0.8m, wrist=0°, gripper=holding_mug
<think><observation>Navigation camera: I am positioned at the counter and successfully holding the target coffee mug as intended. Manipulation camera: The mug is securely grasped in my gripper and clearly visible, confirming successful task completion. Arm state: My arm is extended outward (z=0.2m, y=0.8m), wrist at neutral position, and gripper is successfully holding the target mug.</observation><reasoning>Perfect! I have successfully completed the manipulation task. The target object is now securely in my possession, fulfilling the task requirements.</reasoning><prediction>Task completed successfully. I will maintain this position with the mug securely held, ready for task completion confirmation or any subsequent instructions.</prediction></think>
<answer>Task completed - mug successfully retrieved</answer>
Round 4:
Env_feedback: Success"""
    elif selected_format == "no_think":
        example=f"""Example:
Round 1:
Navigation Camera:
<nav_image>
Manipulation Camera:
<manip_image>
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<answer>moveahead, moveahead</answer>
Round 2:
Env_feedback: Last action is executed successfully.
Navigation Camera:
<nav_image>
Manipulation Camera:
<manip_image>
Arm State: z=0.0m, y=0.8m, wrist=0°, gripper=empty
<answer>move_arm_out, move_arm_out, pickup</answer>
Round 3:
Env_feedback: Last action is executed successfully.
Navigation Camera:
<nav_image>
Manipulation Camera:
<manip_image>
Arm State: z=0.2m, y=0.8m, wrist=0°, gripper=holding_mug
<answer>Task completed</answer>
Round 4:
Env_feedback: Success"""
        
    base_prompt_text = """You are a Stretch robot performing object manipulation tasks in home environments.

DUAL CAMERA SYSTEM:
- Navigation Camera: Provides wide-field view for spatial awareness and navigation
- Manipulation Camera: Provides close-up view of objects within manipulation range

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

COORDINATION STRATEGY:
1. Use Navigation Camera to locate target objects and plan approach
2. Use base movement actions to position yourself optimally
3. Use Manipulation Camera to precisely guide arm movements
4. Monitor arm proprioception to ensure accurate positioning
5. Coordinate both cameras to maintain spatial awareness during manipulation

MANIPULATION WORKFLOW:
1. Navigate to target using navigation camera and base movements
2. Position yourself so target appears in manipulation camera
3. Extend arm outward (move_arm_out) to reach target
4. Adjust arm height (move_arm_up/down) if needed
5. Fine-tune positioning with small movements if necessary
6. Execute pickup when object is within gripper range
7. Use dropoff when you need to release the object

Rewards:
- Format correct: +0.5
- Successful object manipulation: +10.0

The instruction will be provided with each observation. Use both cameras and arm proprioception to complete manipulation tasks efficiently."""
    return base_prompt_text + '\n' + example

# init_observation_template and action_template for SPOC robot manipulation
def init_observation_template(**kwargs):
    observation = kwargs.get("observation", "No observation provided.")
    instruction = kwargs.get("instruction", "No instruction provided.")
    arm_state = kwargs.get("arm_state", "z=0.0m, y=0.8m, wrist=0°, gripper=empty")
    
    return f"""[Initial Observation]:
Navigation Camera:
{observation}
Manipulation Camera:
{observation}
Arm State: {arm_state}
Human Instruction: {instruction}
Decide your next action(s)."""

def action_template(**kwargs):
    observation = kwargs.get("observation", "No observation provided.")
    instruction = kwargs.get("instruction", "No instruction provided.")
    valid_action = kwargs.get("valid_action", "No valid action provided.")
    env_feedback = kwargs.get("env_feedback", "No environment feedback provided.")
    reward = kwargs.get("reward", "No reward provided.")
    done = kwargs.get("done", "No done status provided.")
    arm_state = kwargs.get("arm_state", "z=0.0m, y=0.8m, wrist=0°, gripper=empty")
    
    return f"""After your answer, the extracted valid action is {valid_action}.
The environment feedback is: {env_feedback}
reward: {reward}
done: {done}
After that, the observation is:
Navigation Camera:
{observation}
Manipulation Camera:
{observation}
Arm State: {arm_state}
Human Instruction: {instruction}
Decide your next action(s)."""

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


if __name__ == "__main__":
    # Example usage
    max_actions_per_step = 2
    action_sep = "|"
    
    for key, func in format_prompt.items():
        print(f"{key} format prompt:")
        print(func(max_actions_per_step=max_actions_per_step, action_sep=action_sep))
        print("\n" + "="*50 + "\n")