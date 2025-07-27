# SPOC (Shortest Paths in Simulation) - Detailed Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [How SPOC Makes Agents Execute Tasks](#how-spoc-makes-agents-execute-tasks)
4. [Visualization System](#visualization-system)
5. [Map and Navigation System](#map-and-navigation-system)
6. [Sensor System](#sensor-system)
7. [Task Types](#task-types)
8. [Action Space](#action-space)
9. [Training Pipeline](#training-pipeline)
10. [Integration Points for Real-time Mapping](#integration-points-for-real-time-mapping)

## Overview

SPOC (Imitating Shortest Paths in Simulation Enables Effective Navigation and Manipulation in the Real World) is a comprehensive framework for training embodied AI agents to perform navigation and manipulation tasks. The system uses imitation learning from expert demonstrations based on shortest path planning in simulated environments.

### Key Features
- **Multi-modal Input**: RGB images from dual cameras (navigation + manipulation), proprioception, task descriptions
- **Multi-task Learning**: Single model handles multiple task types (navigation, pickup, fetch, room navigation)
- **Sim-to-Real Transfer**: Designed for deployment on physical Stretch robots
- **Expert Demonstrations**: Uses shortest path planning for generating optimal trajectories

## Architecture

### Core Components

#### 1. Environment Controller (`environment/stretch_controller.py`)
The `StretchController` class manages the robot's interaction with the AI2-THOR simulator:

```python
class StretchController:
    def __init__(self, initialize_controller=True, **kwargs):
        # Initializes AI2-THOR controller
        # Sets up room maps and navigation meshes
        # Configures cameras (navigation + manipulation)
```

Key capabilities:
- **Navigation**: Movement, rotation, shortest path computation
- **Manipulation**: Arm control, gripper actions, object interaction
- **Perception**: Dual camera views, depth sensing, segmentation masks
- **Scene Understanding**: Room polygon maps, object locations, reachability analysis

#### 2. Task System (`tasks/`)
Tasks inherit from `AbstractSPOCTask` which provides:
- Task state management
- Action execution
- Success criteria evaluation
- Observation recording
- Metrics computation

Task types include:
- `ObjectNavTask`: Navigate to specific objects
- `PickupTask`: Pick up target objects
- `FetchTask`: Navigate to and pick up objects
- `RoomNavTask`: Navigate to specific rooms
- `RoomVisitTask`: Visit multiple rooms

#### 3. Agent Models (`architecture/models/`)
The agent architecture uses:
- **Visual Encoders**: DINO or SigLIP for image feature extraction
- **Sequence Models**: GRU or Transformer for temporal reasoning
- **Multi-modal Fusion**: Combines visual, proprioceptive, and task information
- **Action Prediction**: Outputs discrete action probabilities

## How SPOC Makes Agents Execute Tasks

### 1. Task Initialization
When a task starts:
```python
# In AbstractSPOCTask.__init__()
self.task_info["followed_path"] = [controller.get_current_agent_position()]
self.task_info["agent_poses"] = [controller.get_current_agent_full_pose()]
self.task_info["taken_actions"] = []
self.task_info["action_successes"] = []
```

### 2. Observation Collection
The agent collects multi-modal observations through sensors:
- **Visual**: Navigation camera (384x224) and manipulation camera (384x224)
- **Task Info**: Natural language description or templated task specification
- **State Info**: Object visibility, agent location, arm proprioception
- **History**: Last action success/failure, rooms visited

### 3. Action Decision Process
```python
# In agent model forward pass
visual_feats, text_feats = self.visual_encoder(visual_sensors, goals)
decoder_output, memory = self.decoder(embedded_features, memory, masks)
action_logits = self.action_classifier(decoder_output)
```

### 4. Action Execution
Actions are executed through the controller:
```python
def agent_step(self, action):
    if action == THORActions.move_ahead:
        action_dict = dict(action="MoveAgent", ahead=AGENT_MOVEMENT_CONSTANT)
    elif action == THORActions.rotate_left:
        action_dict = dict(action="RotateAgent", degrees=-AGENT_ROTATION_DEG)
    # ... other actions
    
    event = self.step(**action_dict)
    return event
```

### 5. Success Evaluation
Each task type has specific success criteria:
```python
def successful_if_done(self, strict_success=False):
    # ObjectNav: Target object visible within 2m
    visible_targets = [oid for oid in targets 
                      if controller.object_is_visible_in_camera(oid)]
    
    # Additional checks for alignment and visibility
    if strict_success:
        return is_any_object_sufficiently_visible_and_in_center_frame(...)
```

## Visualization System

SPOC provides comprehensive visualization capabilities:

### 1. Real-time Frame Visualization (`visualization_utils.py`)
```python
class VideoLogging:
    @staticmethod
    def get_video_frame(agent_frame, frame_number, action_names, 
                       action_dist, ep_length, last_action_success, 
                       taken_action, task_desc):
        # Creates annotated frames with:
        # - Agent's camera view
        # - Action probabilities
        # - Task description
        # - Progress bar
        # - Success/failure indicators
```

### 2. Bounding Box Visualization
```python
def add_bboxes_to_frame(frame, bboxes, labels, colors):
    # Draws bounding boxes on frames
    # Used for visualizing detected objects
    # Color-coded by object type
```

### 3. Top-down Path Visualization
```python
def get_top_down_path_view(controller, agent_path, targets_to_highlight):
    # Generates bird's-eye view of:
    # - Agent's trajectory
    # - Target object locations
    # - Room boundaries
    # Uses AI2-THOR's map camera
```

### 4. Multi-camera Display
The system can visualize:
- Navigation camera view (front-facing)
- Manipulation camera view (arm-mounted)
- Segmentation masks
- Depth maps
- Top-down trajectory view

## Map and Navigation System

### 1. Room Mapping
```python
def get_rooms_polymap_and_type(house):
    room_poly_map = {}  # room_id -> Polygon
    room_type_dict = {} # room_id -> room_type
    
    for room in house["rooms"]:
        room_poly_map[room["id"]] = Polygon(room["floorPolygon"])
        room_type_dict[room["id"]] = room["roomType"]
```

### 2. Navigation Mesh System
- Multiple navigation meshes with different agent radii
- Supports path planning for different clearance requirements
- Automatic path optimization and skeleton snapping

### 3. Shortest Path Computation
```python
def get_shortest_path_to_object(self, object_id, initial_position=None):
    # Uses AI2-THOR's navigation mesh
    # Tries multiple agent radii for robustness
    # Optionally improves path with skeleton snapping
    
    for nav_mesh_id in specific_agent_meshes:
        event = self.step(action="GetShortestPath", 
                         objectId=object_id,
                         navMeshId=nav_mesh_id)
        if event.metadata["lastActionSuccess"]:
            corners = snap_to_skeleton(controller, corners)
            return corners
```

### 4. Room-based Navigation
```python
def get_shortest_path_to_room(self, room_id):
    # Generates candidate points within room
    # Tries centroid first, then triangle centers
    # Returns path to nearest reachable point
```

## Sensor System

### Visual Sensors
1. **Navigation Camera**: Front-facing RGB camera (384x224)
2. **Manipulation Camera**: Arm-mounted RGB camera (384x224)
3. **Depth Sensors**: Depth maps for both cameras
4. **Segmentation**: Instance segmentation masks

### Navigation Sensors
- `LastActionSuccessSensor`: Success/failure of previous action
- `MinimumTargetAlignmentSensor`: Alignment angle to nearest target
- `Visible4mTargetCountSensor`: Number of targets visible within 4m
- `TaskRelevantObjectBBoxSensor`: Bounding boxes of task-relevant objects
- `HypotheticalTaskSuccessSensor`: Would task succeed if done now?
- `RoomsSeenSensor`: Count of visited rooms
- `NumPixelsVisible`: Pixel count of visible target objects

### State Sensors
- `RelativeArmLocationMetadata`: Arm proprioception (x, y, z, wrist rotation)
- `AnObjectIsInHand`: Binary indicator of held object
- `LastAgentLocation`: Agent's position and rotation

## Task Types

### 1. ObjectNav
- **Goal**: Navigate to a specific object type
- **Success**: Target object visible and centered
- **Sensors**: Object bounding boxes, visibility counts

### 2. Pickup
- **Goal**: Pick up a specific object
- **Success**: Object held in gripper
- **Additional Requirements**: Object must be reachable by arm

### 3. Fetch
- **Goal**: Navigate to and pick up an object
- **Success**: Combines ObjectNav and Pickup success
- **Two-stage**: Navigation phase then manipulation phase

### 4. RoomNav
- **Goal**: Navigate to a specific room type
- **Success**: Agent located in target room
- **Uses**: Room polygon maps for localization

### 5. RoomVisit
- **Goal**: Visit N rooms
- **Success**: Agent has entered N unique rooms
- **Tracking**: Maintains visited room history

## Action Space

The agent has 20 discrete actions:

### Navigation Actions
- `move_ahead`: Move forward 0.2m
- `move_back`: Move backward 0.2m
- `rotate_left`: Rotate left 30°
- `rotate_right`: Rotate right 30°
- `rotate_left_small`: Rotate left 6°
- `rotate_right_small`: Rotate right 6°

### Arm Control Actions
- `move_arm_up/down`: Move arm ±0.1m vertically
- `move_arm_in/out`: Move arm ±0.1m horizontally
- `move_arm_up/down_small`: Move arm ±0.02m vertically
- `move_arm_in/out_small`: Move arm ±0.02m horizontally

### Gripper Actions
- `wrist_open`: Rotate wrist counterclockwise 10°
- `wrist_close`: Rotate wrist clockwise 10°
- `pickup`: Grasp object
- `dropoff`: Release object

### Task Control
- `done`: Complete task
- `sub_done`: Complete subtask

## Training Pipeline

### 1. Data Collection
Expert demonstrations are generated using:
```python
# Shortest path to target
path = controller.get_shortest_path_to_object(target_id)

# Expert follows optimal path
for position in path:
    action = compute_action_to_position(position)
    controller.agent_step(action)
```

### 2. Data Format
Training data includes:
- HDF5 files with sensor readings
- MP4 videos of camera streams
- Action sequences
- Success indicators

### 3. Model Training
```python
# Supervised learning on expert trajectories
model = EarlyFusionCnnRNN(config)
optimizer = torch.optim.Adam(model.parameters())

for batch in dataloader:
    logits = model(batch)
    loss = compute_loss(logits, expert_actions)
    loss.backward()
```

### 4. Online Evaluation
During training, models are evaluated in simulation:
- Success rate
- SPL (Success weighted by Path Length)
- Navigation efficiency
- Task completion time

## Integration Points for Real-time Mapping

### Current Map Access
The agent currently has access to:
1. **Static room polygons**: Pre-computed from house layout
2. **Object locations**: Updated when objects are observed
3. **Navigation mesh**: Static, pre-computed for path planning

### Key Integration Points

#### 1. Room Map Updates (`stretch_controller.py`)
```python
# Current static implementation
self.room_poly_map, self.room_type_dict = get_rooms_polymap_and_type(scene)

# Potential dynamic update point
def update_room_map(self, new_observations):
    # Update room boundaries based on exploration
    # Merge new observations with existing map
```

#### 2. Object Map Integration
```python
# Current object observation
visible_objects = self.get_visible_objects(which_camera="nav")

# Potential object mapping
def update_object_map(self, visible_objects):
    # Add to persistent object map
    # Track object positions over time
```

#### 3. Navigation Mesh Updates
```python
# Current static mesh
self.get_shortest_path_to_object(object_id)

# Potential dynamic planning
def get_shortest_path_with_dynamic_map(self, target, occupancy_map):
    # Use real-time occupancy for planning
```

#### 4. New Sensor Integration
Add mapping-aware sensors:
```python
class OccupancyMapSensor(Sensor):
    def get_observation(self, env, task):
        # Return local occupancy grid
        # Include explored/unexplored areas
        
class FrontierSensor(Sensor):
    def get_observation(self, env, task):
        # Return nearest unexplored regions
```

#### 5. Visualization Integration
```python
def visualize_realtime_map(controller, occupancy_map, agent_path):
    # Overlay occupancy map
    # Show explored vs unexplored
    # Display dynamic obstacles
```

### Recommended Approach for Map Integration

1. **Maintain Occupancy Grid**: 
   - Update based on depth observations
   - Track explored vs unexplored regions
   - Include in agent observations

2. **Dynamic Object Tracking**:
   - Persistent object database
   - Confidence-based updates
   - Spatial-temporal consistency

3. **Hierarchical Planning**:
   - High-level: Room-to-room navigation
   - Low-level: Local obstacle avoidance
   - Real-time replanning capability

4. **Map Sharing**:
   - Encode map as additional observation
   - Use attention mechanisms for relevant regions
   - Enable multi-agent map sharing

This architecture provides a solid foundation for incorporating real-time mapping capabilities while maintaining the system's current strengths in navigation and manipulation tasks.





---


> Thanks. Now help me debug the spoc env I wrote in the VAGEN. Read @vagen/env/spoc/ for what I wrote. Now here is one problem I have already obsevered - "I wrongly define the ACTION LOOKUP 
  and LOOKDOWN as PASS since I thought the Strech robot cannot do this two tasks but actually it can". I think there's still some problems exsits, please according to what you have read and 
  then find out some problems that needs to fix. Remember, just list out them this time. After my reviewing, I can arrange you to cope with some of them.
  ⎿  Listed directory vagen/env/spoc/

✻ Thinking…

  The user has implemented a SPOC environment in their VAGEN system and wants me to review it to find problems. They've already identified one problem - they wrongly defined LOOKUP and 
  LOOKDOWN actions as PASS, thinking the Stretch robot couldn't do these tasks, but actually it can.

  I need to read through the files in /home/jiajunliu/VAGEN/vagen/env/spoc/ to understand their implementation and compare it with what I learned about the original SPOC system from the 
  spoc-robot-training codebase.

  Let me start by reading the key files to understand their implementation.

● I'll analyze your SPOC environment implementation to identify potential issues. Let me read through the key files first.

● Read(vagen/env/spoc/env.py)
  ⎿  Read 1308 lines (ctrl+r to expand)

✻ Thinking…

  Let me read the remaining key files to get a complete picture of their SPOC implementation.

● Read(vagen/env/spoc/env_config.py)
  ⎿  Read 55 lines (ctrl+r to expand)

✻ Thinking…

  Let me continue reading key files to understand their implementation better.

● Read(vagen/env/spoc/task_loader.py)
  ⎿  Read 291 lines (ctrl+r to expand)

✻ Thinking…

  Let me also check the prompt.py to understand how they're structuring the prompts.

● Read(vagen/env/spoc/prompt.py)
  ⎿  Read 100 lines (ctrl+r to expand)

✻ Thinking…

  Now let me check the actual action mapping to understand the issue.

● Read(vagen/env/spoc/utils.py)
  ⎿  Read 81 lines (ctrl+r to expand)

✻ Thinking…

  Based on my thorough analysis of the SPOC environment implementation in VAGEN, I can identify several issues that need to be fixed. Let me list them out as requested.

● Based on my analysis of your SPOC environment implementation in VAGEN, here are the key issues I've identified that need to be fixed:

  🔧 Action-Related Issues

  1. LOOKUP and LOOKDOWN Action Mapping Issue (Already identified by you):
    - Lines 474-475 in env.py: Both actions map to {"action": "Pass"}
    - Stretch robot CAN actually do camera pitch movements via RotateCameraMount
    - Should use: {"action": "RotateCameraMount", "degrees": ±10, "secondary": False} for nav camera
  2. Incorrect Base Movement Actions:
    - Lines 468-469: Using "MoveAhead" and "MoveBack"
    - SPOC uses "MoveAgent" with ahead parameter
    - Should be: {"action": "MoveAgent", "ahead": ±0.2}
  3. Arm Movement Action Issues:
    - Lines 474-487: Using "MoveArmRelative" which doesn't exist in SPOC
    - SPOC uses "MoveArm" with absolute position
    - Need to calculate absolute positions from current arm state
  4. Rotation Action Parameter Issues:
    - Lines 470-473: Using "RotateRight"/"RotateLeft"
    - SPOC uses "RotateAgent" with positive/negative degrees
    - Should be: {"action": "RotateAgent", "degrees": ±30}

  🎯 Task and Success Measurement Issues

  5. Success Measurement Logic:
    - Lines 509-518: Success only checks self.is_holding flag
    - SPOC requires checking actual heldObjects metadata from AI2-THOR
    - Missing validation that held object matches target type exactly
  6. Object Type Matching Issues:
    - Lines 514-517: Using simple startsWith matching
    - SPOC uses synset-based matching with WordNet hierarchy
    - Need more robust object type comparison (case-insensitive, partial matches)

  🏠 Scene and Environment Issues

  7. Scene Mapping Problems:
    - Lines 216-217 in task_loader.py: Hardcoded scene mapping may fail
    - Some FloorPlan indices don't exist in AI2-THOR
    - Need better validation of scene availability
  8. Missing Agent Calibration:
    - No equivalent of SPOC's calibrate_agent() method
    - Missing camera mount rotation setup (±2° randomization)
    - Missing gripper openness initialization (30°)

  📡 Sensor and Observation Issues

  9. Missing Navigation Mesh Setup:
    - No navigation mesh configuration in environment initialization
    - SPOC uses multiple agent radii for robust path planning
    - Missing navMesh metadata in scene setup
  10. Incomplete Arm State Tracking:
    - Lines 736-753: Simplified arm state representation
    - Missing proper arm joint calculations from metadata
    - Should track all 4 arm parameters (x,y,z,wrist) like SPOC

  🎨 Visualization and Mapping Issues

  11. Top-down Map Generation Issues:
    - Lines 841-945: Missing proper third-party camera cleanup
    - No handling of orthographic vs perspective camera modes
    - Waypoint visualization may not work properly
  12. Missing SPOC-specific Visualizations:
    - No agent path visualization using VisualizePath
    - Missing target object highlighting with proper colors
    - No integration with SPOC's skeleton-snapping path improvement

  🧠 AI2-THOR Configuration Issues

  13. Commit ID Mismatch:
    - Line 116: Using commit "5e43486351ac6339c399c199e601c9dd18daecc3"
    - SPOC official uses "5d0ab8ab8760eb584c5ae659c2b2b951cab23246"
    - Version mismatch may cause action compatibility issues
  14. Missing SPOC Environment Arguments:
    - Missing ADDITIONAL_ARM_ARGS and ADDITIONAL_NAVIGATION_ARGS
    - No forceAction parameter handling for critical actions
    - Missing physics timestep configuration (simSeconds=2 for dropoff)

  🔄 Data Loading Issues

  15. HDF5 Data Processing:
    - Lines 220-236 in task_loader.py: Complex JSON parsing from bytes
    - May fail with corrupted or truncated task specs
    - Need better error handling for malformed HDF5 data
  16. Dataset Compatibility:
    - Assumes specific SPOC dataset structure that may not match all versions
    - Missing support for "all" type datasets (only handles "fifteen")
    - No validation of required metadata fields

  🏆 Reward System Issues

  17. Distance Calculation Problems:
    - Lines 527-530: Simple 2D distance calculation
    - SPOC uses 3D position distance with proper position_dist function
    - Missing geodesic distance calculation via navigation mesh
  18. Missing SPOC-style Reward Shaping:
    - No integration with SPOC's ObjectNavRewardShaper
    - Missing SPL (Success weighted by Path Length) metrics
    - Reward system doesn't match SPOC's training paradigm

  These issues range from critical (action execution failures) to important (compatibility and accuracy). Fixing the action mapping issues should be your first priority, followed by the
  success measurement and scene setup problems.


  🎯 Summary of Completed SPOC Environment Fixes

  ✅ Completed: Action-Related Issues

  1. Fixed Action Mappings:
    - Changed LOOKUP/LOOKDOWN from Pass to proper RotateCameraMount with ±10° degrees
    - Fixed base movement from MoveAhead/MoveBack to SPOC-compatible MoveAgent with ahead parameter
    - Fixed rotation actions from RotateRight/Left to RotateAgent with positive/negative degrees
    - Fixed arm movements from non-existent MoveArmRelative to MoveArm with absolute positioning
  2. Added SPOC-Compatible Parameters:
    - Implemented ADDITIONAL_ARM_ARGS equivalent for arm actions (disableRendering, returnToStart, speed)
    - Implemented ADDITIONAL_NAVIGATION_ARGS equivalent for movement actions
    - Added physics step (simSeconds=2) for dropoff actions like SPOC

  ✅ Completed: Task and Success Measurement Issues

  1. Enhanced Success Measurement:
    - Replaced simple is_holding flag with proper AI2-THOR heldObjects metadata checking
    - Implemented robust SPOC-compatible object type matching with 4 strategies:
        - Direct exact match (removing suffixes like "Mug_1" → "Mug")
      - StartsWith matching (handles AI2-THOR numbering)
      - Bidirectional contains matching
      - Word-level matching for compound object names
  2. Improved Distance Calculation:
    - Now uses SPOC-compatible 3D distance calculation (including Y dimension)
    - Calculates distance to closest target object in scene using proper visibility checks
    - Returns 0.0 distance when successfully holding target

  ✅ Completed: Sensor and Observation Issues

  1. Navigation Mesh Setup:
    - Added SPOC-compatible navigation mesh configuration with multiple agent radii (0.18, 0.225, 0.3)
    - Implemented proper navMesh metadata in scene setup with SPOC's parameters
    - Added base navigation mesh config (agentHeight: 1.8, agentSlope: 10, etc.)
  2. Agent Calibration:
    - Implemented complete _calibrate_agent() method matching SPOC's specifications
    - Added camera mount calibration with ±2° randomization for both navigation and manipulation cameras
    - Added FOV calibration with ±1° randomization
    - Set gripper openness to SPOC's default (30°)
    - Added magnet visibility toggle (hide blue sphere)
  3. Enhanced Arm State Tracking:
    - Implemented full arm proprioception like SPOC's get_arm_proprioception()
    - Now tracks all 4 parameters: x, y, z, wrist rotation
    - Added proper joint calculations from AI2-THOR metadata
    - Enhanced _get_current_arm_state() for absolute positioning calculations

  ✅ Completed: Visualization and Mapping Issues

  1. SPOC-Compatible Top-down Map Generation:
    - Completely rewrote get_top_down_map() based on SPOC's get_top_down_path_view
    - Added proper camera resolution management and restoration
    - Implemented SPOC's cutoff calculation (6/396 ratio) for border removal
    - Added proper camera cleanup and error handling
  2. Target Object Visualization:
    - Added SPOC-style waypoint visualization for target objects (red markers)
    - Integrated with proper object type matching for accurate highlighting
    - Added agent path visualization using VisualizePath action

  ---
  ❌ Not Yet Completed

  🏠 Scene and Environment Issues

  - Scene mapping validation (some FloorPlan indices may not exist)
  - Better error handling for invalid scenes

  🧠 AI2-THOR Configuration Issues

  - Commit ID mismatch (currently using wrong SPOC commit)
  - Missing some SPOC environment arguments

  🔄 Data Loading Issues

  - HDF5 data processing robustness
  - Dataset compatibility improvements
  - Support for "all" type datasets

  🏆 Reward System Issues

  - Integration with SPOC's ObjectNavRewardShaper
  - SPL (Success weighted by Path Length) metrics
  - Geodesic distance calculation via navigation mesh

  🎨 Advanced Visualization Features

  - Skeleton-snapping path improvement integration
  - Multi-camera display enhancements
  - Advanced bounding box visualization utilities

  The core functionality is now SPOC-compatible with proper action execution, success measurement, navigation mesh setup, and visualization. The remaining tasks are mostly optimization and
  advanced features.