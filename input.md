# VAGEN Agent Input Structure

This document describes exactly what input the VLM agent receives at each step.

## Overview

The agent receives **full conversation history** through a chat-based interface, with each step containing:
- Previous actions and their feedback
- Current environment observation 
- Visual data (images)
- Task instruction

## Input Format

### 1. System Prompt (Once at start)
```
You are an agent in a simulated environment with a robotic body equipped with navigation and manipulation capabilities. Your goal is to complete household tasks by navigating and manipulating objects.

## Available Actions:
- **moveahead**: Move the agent's base forward by 0.2 meters
- **moveback**: Move the agent's base backward by 0.2 meters  
- **rotateright**: Rotate the agent's base right by 30°
- **rotateleft**: Rotate the agent's base left by 30°
- **rotateright_small**: Rotate the agent's base right by 6°
- **rotateleft_small**: Rotate the agent's base left by 6°
- **pickup**: Initiate a grasp action to pick up an object
- **dropoff**: Execute a release action to drop an object
- **move_arm_up**: Move the arm up by 0.1 meters
- **move_arm_down**: Move the arm down by 0.1 meters
- **move_arm_out**: Extend the arm outward by 0.1 meters
- **move_arm_in**: Retract the arm inward by 0.1 meters
- **wrist_open**: Rotate the wrist counterclockwise by 10°
- **wrist_close**: Rotate the wrist clockwise by 10°
- **move_arm_up_small**: Move the arm up by 0.02 meters
- **move_arm_down_small**: Move the arm down by 0.02 meters
- **move_arm_out_small**: Extend the arm outward by 0.02 meters
- **move_arm_in_small**: Retract the arm inward by 0.02 meters

## Response Format:
You should respond with your reasoning in <think></think> tags, then provide your action choice.

Example:
<think>I need to explore the room to find the target object. I'll start by moving forward.</think>

I'll move forward to explore the environment.

**moveahead**

Decide your next action(s).
```

### 2. Conversation History Pattern

For each step, the conversation follows this pattern:

#### A. Environment Message (User role):
```
After your answer, the extracted valid action is ["rotateleft"].
The environment feedback is: Action 'rotateleft' FAILED - collision - cannot rotate. Try a different approach.
reward: -0.5
done: False
After that, the observation is:
Visual Observation: The robot observes a moderately lit indoor household environment [WARNING: Last action failed due to collision or physical constraints]. The robot is searching for a melon. Continue exploring to find the target object. <image> <image> <image>
Arm State: position=(0.00,0.80,0.00), gripper=empty
Human Instruction: Find and fetch a melon. Navigate around the environment to locate the melon, then approach it and pick it up.
Decide your next action(s).
```

#### B. Agent Response (Assistant role):
```
<think>
My last action to rotate left failed due to collision. The environment is warning me about physical constraints. I should try a different approach to avoid obstacles.
</think>

I need to try a different direction since rotating left failed due to collision.

**rotateright**
```

### 3. Current Step Input Structure

#### Initial Observation (First step):
```
[Initial Observation]:
Visual Observation: The robot observes a moderately lit indoor household environment. The robot is searching for a melon. Continue exploring to find the target object. <image> <image> <image>
Arm State: position=(0.00,0.80,0.00), gripper=empty
Human Instruction: Find and fetch a melon. Navigate around the environment to locate the melon, then approach it and pick it up.
Decide your next action(s).
```

#### Subsequent Steps:
```
After your answer, the extracted valid action is ["{last_action}"].
The environment feedback is: {detailed_feedback}
reward: {reward_value}
done: {true/false}
After that, the observation is:
Visual Observation: {scene_description} <image> <image> <image>
Arm State: {arm_position_and_state}
Human Instruction: {task_instruction}
Decide your next action(s).
```

## Key Input Components

### Environment Feedback Types:
- **Success**: `"Action 'moveahead' executed successfully."`
- **Collision**: `"Action 'rotateleft' FAILED - collision - cannot rotate. Try a different approach."`
- **Arm Limits**: `"Action 'move_arm_up' FAILED - arm at limit or collision. Try a different approach."`
- **Pickup Failure**: `"Action 'pickup' FAILED - no object within reach to pick up. Try a different approach."`

### Visual Observation Structure:
- **Scene Description**: Text describing the environment and any warnings
- **Three Images**: `<image> <image> <image>`
  1. Navigation camera view
  2. Manipulation camera view  
  3. Top-down map with agent path
- **Warning Messages**: Added when actions fail (e.g., `[WARNING: Last action failed due to collision or physical constraints]`)

### Arm State Format:
```
position=(x,y,z), gripper={empty/holding_object}
```

### Reward Information:
- **Positive**: Successful actions, progress toward goal, target visibility
- **Negative**: Failed actions (-0.5), ineffective movements
- **Large**: Task completion (+10.0), picking up target object

### History Management:
- **Current Step**: Full visual data with `<image>` tags
- **Previous Steps**: Text only with `[Previous image omitted]` markers
- **Complete Context**: Agent sees all previous actions and their outcomes

## Action Success Feedback System

The system now provides detailed feedback for failed actions:

### Movement Actions:
- Success: Position changed by >0.01m
- Failure: "collision - cannot move in that direction"

### Rotation Actions:
- Success: Rotation changed by >2°
- Failure: "collision - cannot rotate"

### Arm Actions:
- Success: Arm position changed by >0.001m  
- Failure: "arm at limit or collision"

### Object Manipulation:
- Pickup Success: Successfully grasping an object
- Pickup Failure: "no object within reach to pick up"
- Dropoff Success: Successfully releasing an object
- Dropoff Failure: "failed to release object"

## Example Full Conversation

```
System: [System prompt with actions and instructions]

User: [Initial Observation with 3 images]