#!/usr/bin/env python3
"""
Final SPOC Map Visualization Test - Complete working implementation
This creates a simple fetch task and generates SPOC-style map visualizations during execution
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import random
import matplotlib
matplotlib.use('Agg')
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

class SPOCMapAgent:
    """Simple agent for SPOC map demo."""
    
    def __init__(self, target_object_type="apple"):
        self.target_object_type = target_object_type.lower()
        self.exploration_steps = 0
        self.target_found = False
        self.task_completed = False
        self.last_positions = []
        self.stuck_counter = 0
        
    def get_action(self, controller):
        """Get next action based on current state."""
        
        # Get current state
        metadata = controller.last_event.metadata
        agent_pos = metadata["agent"]["position"]
        
        # Track positions for stuck detection
        pos_key = f"{agent_pos['x']:.1f},{agent_pos['z']:.1f}"
        self.last_positions.append(pos_key)
        if len(self.last_positions) > 5:
            self.last_positions.pop(0)
        
        # Check if stuck
        if len(self.last_positions) >= 3:
            if len(set(self.last_positions[-3:])) <= 1:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        
        # Check if holding target object
        held_objects = metadata.get('arm', {}).get('heldObjects', [])
        for obj in held_objects:
            if self.target_object_type in obj.get('objectType', '').lower():
                self.task_completed = True
                return "done", "Task completed successfully!"
        
        # Find target object in visible objects
        target_object = None
        closest_distance = float('inf')
        
        for obj in metadata.get('objects', []):
            if (obj.get('visible', False) and 
                self.target_object_type in obj.get('objectType', '').lower()):
                
                obj_pos = obj['position']
                distance = ((agent_pos['x'] - obj_pos['x'])**2 + 
                           (agent_pos['z'] - obj_pos['z'])**2)**0.5
                
                if distance < closest_distance:
                    closest_distance = distance
                    target_object = obj
        
        # If target is visible and close, try to pick it up
        if target_object and closest_distance < 1.5:
            self.target_found = True
            if closest_distance < 0.8:
                return "pickup", f"Attempting to pickup {target_object['objectType']}"
            else:
                return "move_toward_target", f"Moving toward {target_object['objectType']}"
        
        # If stuck, try to get unstuck
        if self.stuck_counter > 2:
            self.stuck_counter = 0
            if random.random() < 0.5:
                return "move_back", "Getting unstuck - moving back"
            else:
                return "rotate_random", "Getting unstuck - rotating"
        
        # Exploration behavior
        self.exploration_steps += 1
        
        if self.exploration_steps % 8 == 0:
            return "rotate_right", "Scanning for target object"
        elif self.exploration_steps % 15 == 0:
            return "rotate_left", "Changing exploration direction"
        else:
            return "move_ahead", f"Exploring to find {self.target_object_type}"

def setup_spoc_controller():
    """Setup AI2-THOR controller with SPOC-compatible settings."""
    config = {
        "width": 396,
        "height": 224,
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
    return controller

def execute_action(controller, action_type, info=""):
    """Execute a specific action and return success status."""
    
    action_map = {
        "move_ahead": {"action": "MoveAgent", "ahead": 0.2},
        "move_back": {"action": "MoveAgent", "ahead": -0.2},
        "rotate_left": {"action": "RotateAgent", "degrees": -30},
        "rotate_right": {"action": "RotateAgent", "degrees": 30},
        "rotate_random": {"action": "RotateAgent", "degrees": random.choice([-90, -45, 45, 90])},
        "pickup": {"action": "PickupObject"},
        "move_toward_target": {"action": "MoveAgent", "ahead": 0.15},
        "done": {"action": "Pass"}  # No-op for done
    }
    
    params = action_map.get(action_type, {"action": "Pass"})
    event = controller.step(**params)
    
    success = event.metadata.get('lastActionSuccess', False)
    return success, event

def generate_spoc_map(controller, agent_path, target_objects=None, map_size=(600, 600)):
    """Generate SPOC-style top-down map."""
    
    try:
        # Setup third party camera if needed
        if len(controller.last_event.third_party_camera_frames) < 2:
            event = controller.step(action="GetMapViewCameraProperties")
            if event.metadata.get('lastActionSuccess', False):
                cam = event.metadata["actionReturn"].copy()
                # Increase orthographic size for better view
                if "orthographicSize" in cam:
                    cam["orthographicSize"] += 1
                
                controller.step(
                    action="AddThirdPartyCamera",
                    **cam,
                    skyboxColor="white"
                )
        
        # Get the current map frame
        event = controller.last_event
        if len(event.third_party_camera_frames) > 0:
            map_frame = event.third_party_camera_frames[-1].copy()
            
            # Apply cutoff like SPOC does
            cutoff = round(map_frame.shape[1] * 6 / 396)
            map_frame = map_frame[:, cutoff:-cutoff, :]
            
            # Convert RGBA to RGB if needed
            if map_frame.shape[2] == 4:
                map_frame = map_frame[:, :, :3]
            
            # Convert to PIL for drawing
            map_pil = Image.fromarray(map_frame.astype(np.uint8))
            draw = ImageDraw.Draw(map_pil)
            
            # Get camera properties for coordinate conversion
            cam_props = controller.last_event.metadata.get("thirdPartyCameras", [{}])[0]
            if not cam_props:
                return np.array(map_pil)
            
            # Simple coordinate conversion (this is approximate)
            def world_to_map(pos):
                # This is a simplified conversion - may need adjustment
                x_range = 10  # Approximate world size
                z_range = 10
                
                map_x = int((pos['x'] + x_range/2) / x_range * map_pil.width)
                map_y = int((pos['z'] + z_range/2) / z_range * map_pil.height)
                
                return map_x, map_y
            
            # Draw agent path
            if len(agent_path) > 1:
                path_points = [world_to_map(pos) for pos in agent_path]
                for i in range(len(path_points) - 1):
                    draw.line([path_points[i], path_points[i+1]], fill=(0, 0, 255), width=3)
            
            # Draw target objects
            if target_objects:
                for obj_type in target_objects:
                    for obj in controller.last_event.metadata.get('objects', []):
                        if obj_type.lower() in obj.get('objectType', '').lower():
                            obj_x, obj_y = world_to_map(obj['position'])
                            # Draw red circle for target
                            draw.ellipse(
                                [(obj_x-10, obj_y-10), (obj_x+10, obj_y+10)], 
                                fill=(255, 0, 0), 
                                outline=(200, 0, 0)
                            )
                            break
            
            # Draw current agent position
            if agent_path:
                agent_x, agent_y = world_to_map(agent_path[-1])
                draw.ellipse(
                    [(agent_x-5, agent_y-5), (agent_x+5, agent_y+5)], 
                    fill=(0, 255, 0), 
                    outline=(0, 200, 0)
                )
                
            return np.array(map_pil)
        else:
            return np.full((224, 396, 3), 128, dtype=np.uint8)
            
    except Exception as e:
        print(f"Map generation error: {e}")
        import traceback
        traceback.print_exc()
        return np.full((224, 396, 3), 128, dtype=np.uint8)

def create_enhanced_map_visualization(map_image, step, action, target, agent_info, success=False):
    """Create enhanced map with annotations like SPOC paper."""
    
    # Convert to PIL
    if isinstance(map_image, np.ndarray):
        map_pil = Image.fromarray(map_image.astype(np.uint8))
    else:
        map_pil = map_image
    
    # Create larger canvas
    width, height = map_pil.size
    canvas_width = width + 250
    canvas = Image.new('RGB', (canvas_width, height), color='white')
    canvas.paste(map_pil, (0, 0))
    
    # Add annotations
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.load_default()
        title_font = font
    except:
        font = None
        title_font = None
    
    info_x = width + 15
    info_y = 15
    line_height = 18
    
    # Title
    draw.text((info_x, info_y), "SPOC Navigation Demo", font=title_font, fill='black')
    info_y += line_height * 1.5
    
    # Task info
    draw.text((info_x, info_y), f"Task: Fetch {target}", font=font, fill='blue')
    info_y += line_height
    
    # Step info
    draw.text((info_x, info_y), f"Step: {step}", font=font, fill='black')
    info_y += line_height
    
    # Action info
    action_text = action[:25] + "..." if len(action) > 25 else action
    draw.text((info_x, info_y), f"Action: {action_text}", font=font, fill='purple')
    info_y += line_height * 1.5
    
    # Agent status
    draw.text((info_x, info_y), "Agent Status:", font=font, fill='black')
    info_y += line_height
    
    status_lines = [
        f"Holding: {'Yes' if agent_info.get('holding') else 'No'}",
        f"Target found: {'Yes' if agent_info.get('target_found') else 'No'}",
        f"Exploring: {agent_info.get('exploration_steps', 0)} steps"
    ]
    
    for line in status_lines:
        draw.text((info_x, info_y), line, font=font, fill='gray')
        info_y += line_height
    
    # Success indicator
    if success:
        info_y += line_height
        draw.text((info_x, info_y), "TASK COMPLETE!", font=title_font, fill='green')
    
    # Legend
    legend_y = height - 80
    draw.text((info_x, legend_y), "Legend:", font=font, fill='black')
    legend_y += line_height
    
    # Red dot for target
    draw.ellipse((info_x, legend_y, info_x + 8, legend_y + 8), fill='red')
    draw.text((info_x + 12, legend_y - 2), "Target object", font=font, fill='black')
    legend_y += line_height
    
    # Blue line for path  
    draw.line((info_x, legend_y + 4, info_x + 25, legend_y + 4), fill='blue', width=2)
    draw.text((info_x + 30, legend_y - 2), "Agent path", font=font, fill='black')
    
    return canvas

def run_spoc_map_demo():
    """Run complete SPOC map visualization demo."""
    
    print("=" * 70)
    print("🗺️ SPOC Map Visualization Demo - Interactive Agent Task")
    print("=" * 70)
    
    # Create output directory
    output_dir = "./spoc_map_demo"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Setup controller
        print("🔧 Setting up AI2-THOR controller...")
        controller = setup_spoc_controller()
        print(f"✅ Controller ready with commit: {controller._build.commit_id}")
        
        # Test different scenes and targets
        demo_configs = [
            {"scene": "FloorPlan1", "target": "apple", "max_steps": 25},
            # {"scene": "FloorPlan2", "target": "mug", "max_steps": 25},
            # {"scene": "FloorPlan3", "target": "book", "max_steps": 20}
        ]
        
        for config_idx, config in enumerate(demo_configs):
            scene_name = config["scene"]
            target_object = config["target"]
            max_steps = config["max_steps"]
            
            print(f"\n{'='*50}")
            print(f"📋 Demo {config_idx + 1}/3: {scene_name} - Find {target_object}")
            print(f"{'='*50}")
            
            try:
                # Reset scene
                print(f"🏠 Loading scene {scene_name}...")
                event = controller.reset(scene=scene_name)
                if not event.metadata.get('lastActionSuccess', False):
                    print(f"⚠️ Failed to load {scene_name}, skipping...")
                    continue
                
                # Create agent
                agent = SPOCMapAgent(target_object_type=target_object)
                agent_path = [controller.last_event.metadata["agent"]["position"]]
                all_maps = []
                
                print(f"✅ Scene loaded, starting agent...")
                print(f"🎯 Target: {target_object}")
                
                # Save initial frame and map
                if controller.last_event.frame is not None:
                    frame_pil = Image.fromarray(controller.last_event.frame)
                    frame_pil.save(os.path.join(output_dir, f"{scene_name}_{target_object}_frame_000.png"))
                
                # Generate initial map
                initial_map = generate_spoc_map(controller, agent_path, [target_object])
                
                enhanced_map = create_enhanced_map_visualization(
                    initial_map, 0, "START", target_object,
                    {"holding": False, "target_found": False, "exploration_steps": 0}
                )
                enhanced_map.save(os.path.join(output_dir, f"{scene_name}_{target_object}_map_000.png"))
                all_maps.append(enhanced_map)
                
                print(f"📸 Initial state saved")
                
                # Run agent
                step = 0
                task_complete = False
                
                while step < max_steps and not task_complete:
                    step += 1
                    
                    # Get agent action
                    action_type, action_info = agent.get_action(controller)
                    
                    print(f"  Step {step:2d}: {action_type:15s} - {action_info}")
                    
                    # Execute action
                    success, event = execute_action(controller, action_type, action_info)
                    
                    # Update path
                    current_pos = event.metadata["agent"]["position"]
                    agent_path.append(current_pos)
                    
                    # Check if task completed
                    task_complete = agent.task_completed
                    
                    # Save frame
                    if event.frame is not None:
                        frame_pil = Image.fromarray(event.frame)
                        frame_pil.save(os.path.join(output_dir, f"{scene_name}_{target_object}_frame_{step:03d}.png"))
                    
                    # Generate map
                    step_map = generate_spoc_map(controller, agent_path, [target_object])
                    
                    agent_info = {
                        "holding": len(event.metadata.get('arm', {}).get('heldObjects', [])) > 0,
                        "target_found": agent.target_found,
                        "exploration_steps": agent.exploration_steps
                    }
                    
                    enhanced_map = create_enhanced_map_visualization(
                        step_map, step, action_info, target_object, agent_info, success=task_complete
                    )
                    
                    map_path = os.path.join(output_dir, f"{scene_name}_{target_object}_map_{step:03d}.png")
                    enhanced_map.save(map_path)
                    all_maps.append(enhanced_map)
                    
                    print(f"           Success: {success} | Complete: {task_complete}")
                    
                    if task_complete:
                        print(f"    🎉 Task completed in {step} steps!")
                        break
                
                # Create progression summary
                if len(all_maps) >= 4:
                    print("📊 Creating progression summary...")
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    fig.suptitle(f'SPOC Demo: {scene_name} - Find {target_object}', fontsize=16)
                    
                    indices = [0, len(all_maps)//3, 2*len(all_maps)//3, len(all_maps)-1]
                    titles = ['Start', 'Early exploration', 'Mid-task', 'Final state']
                    
                    for ax, idx, title in zip(axes.flat, indices, titles):
                        ax.imshow(all_maps[idx])
                        ax.set_title(f'{title} (Step {idx})', fontsize=12)
                        ax.axis('off')
                    
                    plt.tight_layout()
                    progression_path = os.path.join(output_dir, f"{scene_name}_{target_object}_progression.png")
                    plt.savefig(progression_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"✅ Progression saved: {progression_path}")
                
                print(f"✅ Demo {scene_name} completed!")
                
            except Exception as e:
                print(f"❌ Error in demo {scene_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Close controller
        print("\n🔧 Closing controller...")
        controller.stop()
        
        # Create final summary
        print(f"\n{'='*70}")
        print("📊 SPOC Map Visualization Demo Complete!")
        print(f"{'='*70}")
        print(f"📁 All outputs saved to: {output_dir}/")
        
        # List all generated files
        files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
        print(f"🖼️ Generated {len(files)} visualization files:")
        
        # Group files by type
        maps = [f for f in files if 'map_' in f and 'progression' not in f]
        frames = [f for f in files if 'frame_' in f]
        progressions = [f for f in files if 'progression' in f]
        
        print(f"   • {len(maps)} map visualizations")
        print(f"   • {len(frames)} first-person frames") 
        print(f"   • {len(progressions)} progression summaries")
        
        if progressions:
            print("\n🎬 Key progression summaries:")
            for prog in progressions:
                print(f"   • {prog}")
        
        print(f"\n✅ Demo completed successfully! Check {output_dir}/ for all visualizations.")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_spoc_map_demo()