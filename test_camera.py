#!/usr/bin/env python3
"""
Camera Quality Test Script for SPOC Environment

This script tests various camera configurations to identify what fixes the blur:
1. Tests different nearClippingPlane values (SPOC uses 0.01, you have 0.15)
2. Tests different camera angles (SPOC uses 33° for manip camera, you have 12°)
3. Tests SPOC's exact calibration sequence
4. Saves images for visual comparison

We'll see empirically what works!
"""

import os
import sys
import numpy as np
from PIL import Image
from pathlib import Path
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from vagen.env.spoc.env import SpocEnv
from vagen.env.spoc.env_config import SpocEnvConfig

def calculate_image_sharpness(image: np.ndarray) -> float:
    """Calculate image sharpness using Laplacian variance."""
    if image is None or image.size == 0:
        return 0.0
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image
    
    # Simple edge detection using gradients
    # Calculate differences in x and y directions
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    
    # Calculate variance of gradients as sharpness metric
    sharpness = np.var(dx) + np.var(dy)
    return sharpness

def apply_spoc_official_calibration(env):
    """Apply SPOC's official camera calibration from their codebase."""
    print("   🎯 Applying SPOC official calibration...")
    
    # Preserve current position
    current_pos = env.env.last_event.metadata['agent']['position']
    current_rot = env.env.last_event.metadata['agent']['rotation']
    
    # Step 1: Teleport with horizon=0 and standing=True
    env.env.step(action="Teleport", position=current_pos, rotation=current_rot, horizon=0, standing=True)
    
    # Step 2: Navigation camera mount (27° ± 2° randomization)
    nav_angle = 27.0 + np.random.uniform(-2, 2)
    env.env.step(
        action="RotateCameraMount",
        degrees=nav_angle,
        secondary=False
    )
    
    # Step 3: Navigation camera FOV (59° ± 1° randomization)
    nav_fov = 59 + np.random.uniform(-1, 1)
    env.env.step(
        action="ChangeFOV",
        fieldOfView=nav_fov,
        camera="FirstPersonCharacter"
    )
    
    # Step 4: Manipulation camera mount (33° ± 2° randomization) - NOT 12°!
    manip_angle = 33.0 + np.random.uniform(-2, 2)
    env.env.step(
        action="RotateCameraMount",
        degrees=manip_angle,
        secondary=True
    )
    
    # Step 5: Manipulation camera FOV (59° ± 1° randomization)
    manip_fov = 59 + np.random.uniform(-1, 1)
    env.env.step(
        action="ChangeFOV",
        fieldOfView=manip_fov,
        camera="SecondaryCamera"
    )
    
    # Step 6: Set gripper openness (only if environment is ready)
    try:
        env.env.step(action="SetGripperOpenness", openness=30)
    except Exception as e:
        print(f"   ⚠️  SetGripperOpenness failed: {e}")
    
    print(f"   Nav: angle={nav_angle:.1f}°, FOV={nav_fov:.1f}°")
    print(f"   Manip: angle={manip_angle:.1f}°, FOV={manip_fov:.1f}°")

def save_test_image(images, test_name, output_dir="camera_tests"):
    """Save a combined view of all three cameras with labels."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create figure with 3 subplots
    labels = ["Navigation Camera", "Manipulation Camera", "Top-down Map"]
    
    # Calculate dimensions
    img_height = 224
    img_width = 384
    padding = 10
    label_height = 30
    
    # Create combined image
    total_width = img_width * 3 + padding * 4
    total_height = img_height + label_height + padding * 2
    
    combined = Image.new('RGB', (total_width, total_height), color='black')
    
    # Add each image with label
    for i, (img, label) in enumerate(zip(images, labels)):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        
        # Resize if needed
        if img.size != (img_width, img_height):
            img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
        
        # Paste image
        x_offset = padding + i * (img_width + padding)
        y_offset = label_height + padding
        combined.paste(img, (x_offset, y_offset))
        
        # Add label (using PIL's built-in font)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(combined)
        draw.text((x_offset + 10, 5), label, fill='yellow')
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{test_name}_{timestamp}.png"
    filepath = Path(output_dir) / filename
    combined.save(filepath, quality=95)
    
    print(f"✅ Saved test image: {filepath}")
    return filepath

def test_camera_configuration(env, config_name, modifications):
    """Test a specific camera configuration and return quality metrics."""
    print(f"\n🔧 Testing configuration: {config_name}")
    print(f"   Modifications: {modifications}")
    
    # Apply modifications
    for key, value in modifications.items():
        if key == "stabilization_passes":
            # Apply stabilization
            for _ in range(value):
                env.env.step(action="Pass")
                time.sleep(0.05)
        elif key == "manipulation_camera_height":
            # Adjust manipulation camera height
            agent_pos = env.env.last_event.metadata["agent"]["position"]
            agent_rot = env.env.last_event.metadata["agent"]["rotation"]
            
            env.env.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=0,
                position={
                    "x": agent_pos["x"],
                    "y": agent_pos["y"] + value,
                    "z": agent_pos["z"]
                },
                rotation={"x": modifications.get("manip_camera_angle", 33.0), "y": agent_rot["y"], "z": 0},
                fieldOfView=59,
                nearClippingPlane=modifications.get("near_clip", 0.01),  # SPOC uses 0.01!
                farClippingPlane=modifications.get("far_clip", 50)
            )
        elif key == "brightness":
            # Adjust lighting
            env.env.step(
                action="RandomizeLighting",
                brightness=value,
                randomizeColor=False
            )
        elif key == "wait_time":
            # Additional wait time
            time.sleep(value)
        elif key == "apply_spoc_calibration":
            # Apply SPOC's official calibration
            apply_spoc_official_calibration(env)
    
    # Give time for everything to settle
    time.sleep(0.1)
    env.env.step(action="Pass")
    
    # Get camera frames
    nav_frame = env.navigation_camera
    manip_frame = env.manipulation_camera
    map_frame = env.get_top_down_map(include_path=False)
    
    # Calculate quality metrics
    metrics = {
        "nav_sharpness": calculate_image_sharpness(nav_frame) if nav_frame is not None else 0,
        "manip_sharpness": calculate_image_sharpness(manip_frame) if manip_frame is not None else 0,
        "nav_brightness": np.mean(nav_frame) if nav_frame is not None else 0,
        "manip_brightness": np.mean(manip_frame) if manip_frame is not None else 0,
        "nav_contrast": np.std(nav_frame) if nav_frame is not None else 0,
        "manip_contrast": np.std(manip_frame) if manip_frame is not None else 0
    }
    
    print(f"   📊 Metrics:")
    print(f"      Nav  - Sharpness: {metrics['nav_sharpness']:.1f}, Brightness: {metrics['nav_brightness']:.1f}, Contrast: {metrics['nav_contrast']:.1f}")
    print(f"      Manip - Sharpness: {metrics['manip_sharpness']:.1f}, Brightness: {metrics['manip_brightness']:.1f}, Contrast: {metrics['manip_contrast']:.1f}")
    
    # Save test image
    images = [nav_frame, manip_frame, map_frame]
    save_test_image(images, config_name)
    
    return metrics, images

def test_agent_position(env):
    """Test if agent is in a valid indoor position."""
    # Get current frame
    frame = env.env.last_event.frame
    if frame is None:
        return False, "No frame available"
    
    # Check multiple indicators
    brightness = np.mean(frame)
    
    # Check for reachable positions
    reach_event = env.env.step(action="GetReachablePositions")
    reachable_count = len(reach_event.metadata.get('actionReturn', [])) if reach_event.metadata.get('lastActionSuccess') else 0
    
    # Check visible objects
    visible_objects = env.env.last_event.metadata.get('objects', [])
    indoor_objects = [obj for obj in visible_objects if obj.get('visible') and 
                      any(indoor in obj.get('objectType', '').lower() for indoor in 
                          ['table', 'chair', 'sofa', 'bed', 'cabinet', 'shelf', 'desk', 'counter'])]
    
    # Determine if indoor
    is_indoor = brightness > 50 and (reachable_count > 10 or len(indoor_objects) > 0)
    
    info = f"brightness={brightness:.1f}, reachable_positions={reachable_count}, indoor_objects={len(indoor_objects)}"
    return is_indoor, info

def find_valid_indoor_position(env, max_attempts=10):
    """Find a valid indoor position for the agent."""
    print("\n🔍 Searching for valid indoor position...")
    
    # Get reachable positions
    reach_event = env.env.step(action="GetReachablePositions")
    if not reach_event.metadata.get('lastActionSuccess'):
        return None
    
    positions = reach_event.metadata.get('actionReturn', [])
    if not positions:
        return None
    
    # Try center positions first (more likely to be indoors)
    x_coords = [p['x'] for p in positions]
    z_coords = [p['z'] for p in positions]
    center_x = (min(x_coords) + max(x_coords)) / 2
    center_z = (min(z_coords) + max(z_coords)) / 2
    
    # Sort positions by distance from center
    positions_sorted = sorted(positions, 
                            key=lambda p: (p['x'] - center_x)**2 + (p['z'] - center_z)**2)
    
    # Try positions
    for i, pos in enumerate(positions_sorted[:max_attempts]):
        print(f"  Testing position {i+1}/{min(len(positions_sorted), max_attempts)}...")
        
        # Teleport to position
        teleport_event = env.env.step(
            action="TeleportFull",
            position=pos,
            rotation={'x': 0, 'y': 0, 'z': 0},
            horizon=0,
            standing=True,
            forceAction=True
        )
        
        if teleport_event.metadata.get('lastActionSuccess'):
            # Test if indoor
            is_indoor, info = test_agent_position(env)
            print(f"    {info}")
            
            if is_indoor:
                print(f"  ✅ Found valid indoor position: {pos}")
                return pos
    
    return None

def main():
    print("🎥 SPOC Camera Quality Test Script")
    print("=" * 50)
    
    # Create environment
    config = SpocEnvConfig(
        data_path="/home/jiajunliu/spoc_data/fifteen",
        task_type="FetchType",
        chores_split="val",
        prompt_format="grounding_worldmodeling",
        max_actions_per_step=1,
        action_sep=","
    )
    
    print("🌍 Creating SPOC environment...")
    env = SpocEnv(config)
    
    # Reset to a known good scene (avoid problematic ones)
    print("🔄 Resetting environment...")
    
    # Try to use a specific seed for consistency
    obs, info = env.reset(seed=42)
    
    print(f"📍 Scene: {env.episode_data.get('scene', 'unknown')}")
    print(f"🎯 Target: {env.episode_data.get('targetObjectType', 'unknown')}")
    
    # Test current position
    is_indoor, pos_info = test_agent_position(env)
    print(f"\n📍 Current position: {pos_info}")
    
    if not is_indoor:
        print("⚠️  Agent is outside! Looking for indoor position...")
        valid_pos = find_valid_indoor_position(env)
        
        if valid_pos:
            # Apply the position
            env.env.step(
                action="TeleportFull",
                position=valid_pos,
                rotation={'x': 0, 'y': 0, 'z': 0},
                horizon=0,
                standing=True,
                forceAction=True
            )
            
            # Re-test
            is_indoor, pos_info = test_agent_position(env)
            print(f"📍 New position: {pos_info}")
    
    # Test configurations based on SPOC official implementation
    test_configs = [
        ("baseline", {}),  # Current default settings
        
        ("spoc_official_calibration", {
            "apply_spoc_calibration": True,
            "near_clip": 0.01  # CRITICAL: SPOC uses 0.01
        }),
        
        ("critical_near_plane_fix", {
            "manipulation_camera_height": 0.8,
            "near_clip": 0.01,  # CRITICAL: Change from 0.15 to 0.01
            "manip_camera_angle": 33.0  # SPOC uses 33° not 12°
        }),
        
        ("spoc_with_higher_camera", {
            "apply_spoc_calibration": True,
            "manipulation_camera_height": 1.0,
            "near_clip": 0.01
        }),
        
        ("extreme_near_plane", {
            "manipulation_camera_height": 0.8,
            "near_clip": 0.001,  # Even more extreme
            "manip_camera_angle": 33.0
        }),
        
        ("spoc_full_setup", {
            "apply_spoc_calibration": True,
            "manipulation_camera_height": 0.8,
            "near_clip": 0.01,
            "brightness": (0.9, 1.1),
            "stabilization_passes": 5
        })
    ]
    
    # Run tests
    results = {}
    print("\n🧪 Running camera tests...")
    
    for config_name, modifications in test_configs:
        try:
            metrics, images = test_camera_configuration(env, config_name, modifications)
            results[config_name] = metrics
            
            # Small delay between tests
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ Error in {config_name}: {e}")
            results[config_name] = {"error": str(e)}
    
    # Summary
    print("\n📊 Test Summary:")
    print("=" * 50)
    
    best_nav_config = max(results.items(), 
                         key=lambda x: x[1].get('nav_sharpness', 0) if 'error' not in x[1] else -1)
    best_manip_config = max(results.items(),
                           key=lambda x: x[1].get('manip_sharpness', 0) if 'error' not in x[1] else -1)
    
    print(f"🏆 Best navigation camera config: {best_nav_config[0]} (sharpness: {best_nav_config[1].get('nav_sharpness', 0):.1f})")
    print(f"🏆 Best manipulation camera config: {best_manip_config[0]} (sharpness: {best_manip_config[1].get('manip_sharpness', 0):.1f})")
    
    print("\n✅ Test complete! Check the 'camera_tests' directory for output images.")
    print("📁 You can compare the images to see which configuration produces the clearest cameras.")
    
    print("\n" + "="*60)
    print("📊 WHAT TO CHECK:")
    print("="*60)
    print("1. Compare 'baseline' vs 'critical_near_plane_fix' - does nearPlane=0.01 help?")
    print("2. Compare images with 12° vs 33° manipulation camera angle")
    print("3. Check if 'spoc_official_calibration' produces clearer images")
    print("\nPlease let me know which images look clearest!")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()