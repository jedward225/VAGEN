# SPOC HDF5 Dataset Analysis Report

## Executive Summary

This report documents the analysis of SPOC (Shortest Path in Simulation) HDF5 dataset files and identifies a critical bug in the agent pose initialization that causes agents to be teleported with incorrect orientation.

## Dataset Structure

### File Organization
- Location: `/home/jiajunliu/spoc_data/all/`
- Format: HDF5 files named `hdf5_sensors.hdf5`
- Episodes are stored as groups with numeric keys ('0', '1', '2', etc.)

### Available Sensors per Episode
Each episode contains the following sensor data:

| Sensor Name | Shape | Description |
|-------------|-------|-------------|
| `an_object_is_in_hand` | (T, 1) | Whether object is held |
| `house_index` | (T,) | House/scene identifier |
| `hypothetical_task_success` | (T, 1) | Task success if done |
| `last_action_is_random` | (T, 1) | Random action flag |
| `last_action_str` | (T, 200) | Action string (uint8) |
| `last_action_success` | (T, 1) | Action success flag |
| `last_agent_location` | (T, 6) | Agent pose over time |
| `minimum_l2_target_distance` | (T, 1) | Distance to target |
| `minimum_visible_target_alignment` | (T, 1) | Target alignment |
| `relative_arm_location_metadata` | (T, 4) | Arm proprioception |
| `room_current_seen` | (T,) | Room visibility |
| `rooms_seen` | (T,) | Room count |
| `templated_task_spec` | (T, 130) | Task specification |
| `visible_target_4m_count` | (T, 1) | Nearby targets |

Where T = number of timesteps in the trajectory.

## Critical Finding: Agent Pose Data Format

### The Bug
The `last_agent_location` field contains 6 values per timestep, but the current code misinterprets their meaning:

```python
# Current INCORRECT interpretation:
position[0-2]: x, y, z coordinates ✓ (correct)
position[3]: rotation (yaw) ✗ (actually always -0.00)
position[4]: horizon (pitch) ✗ (actually rotation/yaw in degrees)
position[5]: unknown ✗ (actually always 0.00, likely horizon)
```

### Actual Data Format
Based on analysis of multiple episodes:

| Column | Actual Meaning | Range | Evidence |
|--------|----------------|-------|----------|
| 0 | X position | Variable | Changes during navigation |
| 1 | Y position | ~0.90 | Constant (agent height) |
| 2 | Z position | Variable | Changes during navigation |
| 3 | Unused | -0.00 | Always -0.00 |
| 4 | **Rotation (yaw)** | 0-360° | Varies like compass heading |
| 5 | Unused/Horizon | 0.00 | Always 0.00 |

### Example Data

**Episode 0:**
- First position: `[9.00, 0.90, 10.30, -0.00, 150.00, 0.00]`
- Interpreted as: horizon=150° (looking up at ceiling!)
- Should be: rotation=150° (facing southeast), horizon=0°

**Episode 1:**
- First position: `[2.10, 0.90, 4.15, -0.00, 120.00, 0.00]`
- Interpreted as: horizon=120° (looking up)
- Should be: rotation=120° (facing ESE), horizon=0°

## Impact

This misinterpretation causes:
1. Agents spawn looking at the ceiling (horizon 120-270°)
2. Rotation is always set to -0.00 instead of proper heading
3. Navigation becomes difficult as agents can't see the environment

## Recommended Fix

Update the pose parsing in `/home/jiajunliu/VAGEN/vagen/env/spoc/task_loader.py`:

```python
# Line 252-256 should be:
agent_pose = {
    "position": {"x": initial_pose_data[0], "y": initial_pose_data[1], "z": initial_pose_data[2]},
    "rotation": initial_pose_data[4],  # Use column 4 for rotation
    "horizon": 0,  # Keep horizon at 0 (looking straight)
}
```

## Additional Observations

1. **Trajectory Length**: Episodes vary from 13 to 170 timesteps
2. **Starting Positions**: Agents start at various locations in the scene
3. **Task Types**: Include RoomNav, FetchType, etc.
4. **House Indices**: Range from 801+ (need mapping to AI2-THOR FloorPlans)

## Conclusion

The SPOC dataset stores agent trajectories with rotation in column 4 of `last_agent_location`, but the current implementation incorrectly treats this as the horizon angle, causing agents to spawn looking upward instead of forward. This simple fix will resolve the teleportation orientation issue.