#!/usr/bin/env python3
"""
Test SPOC proper top-down visualization
This creates a simple scene and generates a proper bird's-eye view map
"""

import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set environment variables for headless mode
os.environ['FORCE_HEADLESS'] = '1'
os.environ['AI2THOR_ALLOW_RESOLUTION_MISMATCH'] = '1'

env_vars_to_set = {
    'XAUTHORITY': '',
    'XDG_RUNTIME_DIR': '/tmp',
    'GALLIUM_DRIVER': 'softpipe',
    'MESA_GL_VERSION_OVERRIDE': '3.3',
    'LIBGL_ALWAYS_SOFTWARE': '1',
    'LIBGL_ALWAYS_INDIRECT': '1',
    'EGL_PLATFORM': 'surfaceless',
    'PYOPENGL_PLATFORM': 'osmesa',
    'MESA_NO_ERROR': '1',
}
for key, value in env_vars_to_set.items():
    os.environ[key] = value

import ai2thor.controller
import ai2thor.platform
import ai2thor.fifo_server

def create_simple_top_down_map():
    """Create a simple test to verify top-down map generation."""
    
    print("=== SPOC Top-Down Map Test ===")
    
    # Create output directory
    output_dir = "./spoc_proper_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup controller
    print("Setting up controller...")
    config = {
        "width": 600,
        "height": 600,
        "gridSize": 0.25,
        "agentMode": "stretch",
        "commit_id": "5d0ab8ab8760eb584c5ae659c2b2b951cab23246",
        "server_class": ai2thor.fifo_server.FifoServer,
        "platform": ai2thor.platform.CloudRendering,
        "gpu_device": 0,
        "server_timeout": 1000,
        "visibilityDistance": 1.0,
        "fieldOfView": 59,
    }
    
    controller = ai2thor.controller.Controller(**config)
    
    try:
        # Reset to a simple scene
        print("Loading scene FloorPlan1...")
        event = controller.reset(scene="FloorPlan1")
        
        # Save first-person view
        if event.frame is not None:
            Image.fromarray(event.frame).save(
                os.path.join(output_dir, "first_person_view.png")
            )
            print("✓ Saved first-person view")
        
        # Get map view camera properties
        print("Setting up map camera...")
        event = controller.step(action="GetMapViewCameraProperties")
        
        if event.metadata.get('lastActionSuccess', False):
            cam = event.metadata["actionReturn"].copy()
            print(f"Original camera properties: orthographic={cam.get('orthographic')}")
            
            # Force orthographic view for proper top-down
            cam["orthographic"] = True
            
            # Add third party camera
            event = controller.step(
                action="AddThirdPartyCamera",
                **cam,
                skyboxColor="white"
            )
            
            if event.metadata.get('lastActionSuccess', False):
                print("✓ Third party camera added successfully")
                
                # Get the map frame
                if len(event.third_party_camera_frames) > 0:
                    map_frame = event.third_party_camera_frames[-1]
                    
                    # Convert RGBA to RGB if needed
                    if map_frame.shape[2] == 4:
                        map_frame = map_frame[:, :, :3]
                    
                    # Save the raw map
                    Image.fromarray(map_frame).save(
                        os.path.join(output_dir, "raw_top_down_map.png")
                    )
                    print(f"✓ Saved raw top-down map (shape: {map_frame.shape})")
                    
                    # Now test with agent path
                    agent_pos = event.metadata["agent"]["position"]
                    agent_path = []
                    
                    # Simulate a simple path
                    print("Simulating agent movement...")
                    for i in range(5):
                        controller.step(action="MoveAgent", ahead=0.5)
                        controller.step(action="RotateAgent", degrees=30)
                        new_pos = controller.last_event.metadata["agent"]["position"]
                        agent_path.append(new_pos)
                    
                    # Add path visualization
                    print("Visualizing path...")
                    event = controller.step(
                        action="VisualizePath",
                        positions=agent_path,
                        pathWidth=0.05
                    )
                    
                    # Get map with path
                    if len(event.third_party_camera_frames) > 0:
                        map_with_path = event.third_party_camera_frames[-1]
                        
                        if map_with_path.shape[2] == 4:
                            map_with_path = map_with_path[:, :, :3]
                        
                        Image.fromarray(map_with_path).save(
                            os.path.join(output_dir, "map_with_path.png")
                        )
                        print("✓ Saved map with path")
                        
                        # Hide path
                        controller.step(action="HideVisualizedPath")
                    
                    # Test waypoints for objects
                    print("Adding object waypoints...")
                    objects = controller.last_event.metadata.get('objects', [])
                    waypoints = []
                    
                    # Find some objects to highlight
                    for obj in objects[:3]:  # First 3 objects
                        waypoints.append({
                            "position": obj["position"],
                            "color": {"r": 1, "g": 0, "b": 0, "a": 1},
                            "radius": 0.3,
                            "text": obj["objectType"]
                        })
                    
                    if waypoints:
                        event = controller.step(
                            action="VisualizeWaypoints",
                            waypoints=waypoints
                        )
                        
                        if len(event.third_party_camera_frames) > 0:
                            map_with_objects = event.third_party_camera_frames[-1]
                            
                            if map_with_objects.shape[2] == 4:
                                map_with_objects = map_with_objects[:, :, :3]
                            
                            Image.fromarray(map_with_objects).save(
                                os.path.join(output_dir, "map_with_objects.png")
                            )
                            print("✓ Saved map with object markers")
                    
                    # Create comparison figure
                    print("Creating comparison figure...")
                    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                    
                    # Load saved images
                    first_person = Image.open(os.path.join(output_dir, "first_person_view.png"))
                    raw_map = Image.open(os.path.join(output_dir, "raw_top_down_map.png"))
                    path_map = Image.open(os.path.join(output_dir, "map_with_path.png"))
                    obj_map = Image.open(os.path.join(output_dir, "map_with_objects.png"))
                    
                    axes[0, 0].imshow(first_person)
                    axes[0, 0].set_title("First-Person View")
                    axes[0, 0].axis('off')
                    
                    axes[0, 1].imshow(raw_map)
                    axes[0, 1].set_title("Raw Top-Down Map")
                    axes[0, 1].axis('off')
                    
                    axes[1, 0].imshow(path_map)
                    axes[1, 0].set_title("Map with Agent Path")
                    axes[1, 0].axis('off')
                    
                    axes[1, 1].imshow(obj_map)
                    axes[1, 1].set_title("Map with Object Markers")
                    axes[1, 1].axis('off')
                    
                    plt.suptitle("SPOC Visualization Test Results", fontsize=16)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=150)
                    print("✓ Saved comparison figure")
                    
                else:
                    print("✗ No third party camera frames available")
            else:
                print("✗ Failed to add third party camera")
        else:
            print("✗ Failed to get map view camera properties")
        
        print(f"\n✅ Test complete! Check {output_dir}/ for results")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        controller.stop()
        print("Controller stopped")

if __name__ == "__main__":
    create_simple_top_down_map()