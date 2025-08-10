#!/usr/bin/env python3
"""
Clean test script for GPT-4o-mini on SPOC tasks.
Properly uses the SPOC environment's prompt and observation system.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Add VAGEN to path
sys.path.append('/home/jiajunliu/VAGEN')

from vagen.env.spoc.env import SpocEnv
from vagen.env.spoc.env_config import SpocEnvConfig
from vagen.inference.model_interface.openai.model import OpenAIModelInterface
from vagen.inference.model_interface.openai.model_config import OpenAIModelConfig

# ============================================
# Configuration
# ============================================

API_KEY = os.environ.get("OPENAI_API_KEY", "")
SPOC_DATA_PATH = "/home/jiajunliu/spoc_data/all"
OUTPUT_DIR = Path("/home/jiajunliu/VAGEN/results/gpt4o_mini_spoc_clean")

# Test settings
TEST_CONFIG = {
    "task_types": ["ObjectNavType"],  # Can add "FetchType", "PickupType" later
    "episodes_per_task": 1,  # Start with just 1 episode for testing
    "max_steps": 100,  # Very short test first
    "save_images": True,
    "image_interval": 1,  # Save every step
    "debug_mode": True
}

# Model settings
MODEL_CONFIG = {
    "model_name": "gpt-4o-mini",
    "max_tokens": 200,  # Only need actions, not long explanations
    "temperature": 0.3,
    "seed": 42
}

# ============================================
# Setup
# ============================================

def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(OUTPUT_DIR / 'test.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_environment():
    """Set required environment variables."""
    os.environ["SPOC_DATA_PATH"] = SPOC_DATA_PATH
    # Fix: These paths need to be set properly
    os.environ.setdefault("OBJAVERSE_HOUSES_DIR", "/home/jiajunliu/objaverse_houses/houses_2023_07_28")
    os.environ.setdefault("OBJAVERSE_DATA_DIR", "/home/jiajunliu/objaverse_data")
    os.environ["AI2THOR_DISABLE_PROGRESSIVE_LOADING"] = "1"

# ============================================
# Episode Runner
# ============================================

class SPOCEpisodeRunner:
    """Runs a single SPOC episode with GPT-4o-mini."""
    
    def __init__(self, env: SpocEnv, model: OpenAIModelInterface, 
                 task_type: str, episode_id: int, logger: logging.Logger):
        self.env = env
        self.model = model
        self.task_type = task_type
        self.episode_id = episode_id
        self.logger = logger
        self.episode_dir = OUTPUT_DIR / f"{task_type}_{episode_id:03d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.target_object_type = None  # Will be set after reset
        
    def run(self) -> Dict[str, Any]:
        """Run a single episode."""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Episode {self.episode_id} | Task: {self.task_type}")
        self.logger.info(f"{'='*50}")
        
        # Reset environment
        try:
            obs, info = self.env.reset(seed=self.episode_id)
            instruction = info.get('instruction', 'Navigate to target')
            self.logger.info(f"Instruction: {instruction}")
            
            # Extract target object type from instruction for coordinate tracking
            # e.g., "Find and fetch a bust" -> "bust"
            instruction_lower = instruction.lower()
            common_objects = ['bust', 'apple', 'mug', 'book', 'laptop', 'bottle', 'plate', 'bowl', 'cup', 'knife', 'fork', 'spoon']
            for obj_type in common_objects:
                if obj_type in instruction_lower:
                    self.target_object_type = obj_type
                    break
            if not self.target_object_type:
                # Fallback: try to extract from any word after "a " or "the "
                import re
                match = re.search(r'\b(?:a|the)\s+(\w+)', instruction_lower)
                if match:
                    self.target_object_type = match.group(1)
            
            self.logger.info(f"Target object type: {self.target_object_type}")
            
        except Exception as e:
            self.logger.error(f"Reset failed: {e}")
            return self._create_failed_result(str(e))
        
        # Initialize tracking
        results = {
            "task_type": self.task_type,
            "episode_id": self.episode_id,
            "instruction": instruction,
            "steps": [],
            "total_reward": 0.0,
            "success": False,
            "final_metrics": {}
        }
        
        # Get system prompt from environment (it's a method, need to call it)
        system_prompt = self.env.system_prompt()
        
        # Run episode
        done = False
        step = 0
        total_reward = 0.0
        
        while not done and step < TEST_CONFIG["max_steps"]:
            # Get observation from environment (already properly formatted)
            user_message = obs["obs_str"]
            
            # Get model response
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message, 
                     "multi_modal_data": obs.get("multi_modal_data", {})}
                ]
                
                responses = self.model.generate([messages])
                model_response = responses[0]["text"]
                
                # Extract actions if using proper format
                actions = self._extract_actions(model_response)
                self.logger.debug(f"Step {step} | Actions: {actions}")
                
                # Save images with full context (before action)
                if TEST_CONFIG["save_images"] and (step % TEST_CONFIG["image_interval"] == 0 or step < 3):
                    images = obs.get("multi_modal_data", {}).get("<image>", [])
                    if len(images) >= 3:
                        step_info_for_image = {
                            "episode_id": self.episode_id,
                            "step": step,
                            "task_type": self.task_type,
                            "instruction": instruction,
                            "arm_state": info.get('arm_state', 'z=0.0m, y=0.8m, gripper=empty'),
                            "actions": actions,  # Extracted action only
                            "reward": 0,  # Before action execution
                            "success": info.get('task_success', 0)
                        }
                        
                        # Debug: Verify action vs raw response
                        print(f"[STEP DEBUG] Step {step}: action='{actions}' vs raw_len={len(model_response)}")
                        if len(actions) > 100:  # If action is suspiciously long
                            print(f"[STEP DEBUG] WARNING: Action seems to be raw response!")
                            print(f"[STEP DEBUG] Action: {actions[:100]}...")
                        
                        combined = self._create_combined_image(
                            images[0], images[1], images[2],
                            step_info=step_info_for_image,
                            gpt_response=model_response,
                            user_prompt=user_message
                        )
                        combined.save(self.episode_dir / f"step_{step:03d}_detailed.png")
                        self.logger.debug(f"Saved detailed image: step_{step:03d}_detailed.png")
                
            except Exception as e:
                self.logger.error(f"Model error: {e}")
                model_response = "moveahead"  # Fallback
                actions = "moveahead"
            
            # Execute action
            try:
                obs, reward, done, info = self.env.step(model_response)
                total_reward += reward
                
                # Log step results
                self.logger.info(f"Step {step:3d} | Reward: {reward:6.3f} | "
                               f"Total: {total_reward:6.3f} | "
                               f"Success: {info.get('task_success', 0):.1f}")
                
                # Save step data
                results["steps"].append({
                    "step": step,
                    "actions": actions,
                    "reward": reward,
                    "total_reward": total_reward,
                    "success": info.get("task_success", 0),
                    "info": {k: v for k, v in info.items() 
                            if k not in ["observation", "multi_modal_data"]}
                })
                
                # Check for success
                if info.get("task_success", 0) >= 1.0:
                    self.logger.info("✅ Task completed successfully!")
                    results["success"] = True
                    done = True
                    
            except Exception as e:
                self.logger.error(f"Step execution failed: {e}")
                break
            
            step += 1
        
        # Finalize results
        results["total_steps"] = step
        results["total_reward"] = total_reward
        results["final_metrics"] = {
            "spl": info.get("spl", 0) if "info" in locals() else 0,
            "success": results["success"],
            "efficiency": total_reward / max(step, 1)
        }
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _extract_actions(self, response: str) -> str:
        """Extract actions from model response."""
        if "<answer>" in response and "</answer>" in response:
            start = response.find("<answer>") + 8
            end = response.find("</answer>")
            action = response[start:end].strip()
            # Debug what we extracted
            print(f"[ACTION DEBUG] Extracted action: '{action}' from response length: {len(response)}")
            return action
        # Fallback: return first line that looks like an action
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('<') and len(line) < 50:  # Likely an action
                print(f"[ACTION DEBUG] Fallback action: '{line}' from response length: {len(response)}")
                return line
        # Last resort
        print(f"[ACTION DEBUG] No action found, using 'moveahead' from response length: {len(response)}")
        return "moveahead"
    
    def _get_coordinates_info(self) -> Dict[str, Any]:
        """Extract agent and target coordinates from AI2-THOR metadata."""
        try:
            metadata = self.env.controller.last_event.metadata
            coord_info = {
                "agent_pos": None,
                "agent_rot": None,
                "target_objects": [],
                "distance": 0.0
            }
            
            # Get agent position and rotation
            if 'agent' in metadata:
                agent_data = metadata['agent']
                coord_info["agent_pos"] = agent_data['position']  # {x, y, z}
                coord_info["agent_rot"] = agent_data['rotation']['y']  # yaw angle
            
            # Get target object positions
            if self.target_object_type and 'objects' in metadata:
                target_type = self.target_object_type.lower()
                for obj in metadata['objects']:
                    obj_type = obj['objectType'].lower()
                    if (target_type in obj_type or obj_type in target_type or 
                        any(word in obj_type for word in target_type.split()) or
                        any(word in target_type for word in obj_type.split())):
                        coord_info["target_objects"].append({
                            "id": obj['objectId'],
                            "type": obj['objectType'],
                            "position": obj['position'],
                            "visible": obj.get('visible', False)
                        })
            
            # Calculate distance to closest target
            if coord_info["agent_pos"] and coord_info["target_objects"]:
                agent_pos = coord_info["agent_pos"]
                min_dist = float('inf')
                for target in coord_info["target_objects"]:
                    target_pos = target["position"]
                    dist = ((agent_pos['x'] - target_pos['x'])**2 + 
                           (agent_pos['z'] - target_pos['z'])**2)**0.5
                    min_dist = min(min_dist, dist)
                coord_info["distance"] = min_dist if min_dist != float('inf') else 0.0
            
            return coord_info
            
        except Exception as e:
            self.logger.warning(f"Failed to extract coordinates: {e}")
            return {"agent_pos": None, "agent_rot": None, "target_objects": [], "distance": 0.0}
    
    def _create_combined_image(self, nav_img, manip_img, map_img, 
                              step_info: Dict = None, gpt_response: str = None, 
                              user_prompt: str = None) -> Image.Image:
        """Create detailed combined image with annotations like the original style."""
        print(f"[IMAGE DEBUG] Creating detailed image with step_info: {step_info is not None}")
        print(f"[IMAGE DEBUG] GPT response length: {len(gpt_response) if gpt_response else 0}")
        print(f"[IMAGE DEBUG] User prompt length: {len(user_prompt) if user_prompt else 0}")
        
        # Convert to RGB 
        nav_img = nav_img.convert('RGB') if nav_img else Image.new('RGB', (384, 224), 'gray')
        manip_img = manip_img.convert('RGB') if manip_img else Image.new('RGB', (384, 224), 'gray')
        map_img = map_img.convert('RGB') if map_img else Image.new('RGB', (396, 224), 'gray')
        
        # Resize to display size
        img_size = (400, 400)
        nav_display = nav_img.resize(img_size, Image.Resampling.LANCZOS)
        manip_display = manip_img.resize(img_size, Image.Resampling.LANCZOS)
        map_display = map_img.resize(img_size, Image.Resampling.LANCZOS)
        
        # Create large canvas for text
        padding = 20
        text_height = 800  # Even more space for full prompt and response
        canvas_width = img_size[0] * 3 + padding * 4
        canvas_height = img_size[1] + text_height + padding * 3
        
        print(f"[IMAGE DEBUG] Canvas size: {canvas_width} x {canvas_height}")
        
        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        draw = ImageDraw.Draw(canvas)
        
        # Draw red border for debugging
        draw.rectangle([0, 0, canvas_width-1, canvas_height-1], outline='red', width=2)
        
        # Paste images at top
        canvas.paste(nav_display, (padding, padding))
        canvas.paste(manip_display, (img_size[0] + padding * 2, padding))
        canvas.paste(map_display, (img_size[0] * 2 + padding * 3, padding))
        
        # Load fonts
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            normal_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            title_font = ImageFont.load_default()
            normal_font = title_font
            small_font = title_font
        
        # Draw labels under images
        label_y = padding + img_size[1] + 10
        draw.text((padding, label_y), "Navigation Camera", fill='black', font=title_font)
        draw.text((img_size[0] + padding * 2, label_y), "Manipulation Camera", fill='black', font=title_font)
        draw.text((img_size[0] * 2 + padding * 3, label_y), "Top-Down Map", fill='black', font=title_font)
        
        # Episode info
        info_start_y = label_y + 40
        current_y = info_start_y
        
        if step_info:
            draw.text((padding, current_y), "Episode Information:", fill='blue', font=title_font)
            current_y += 25
            
            info_lines = [
                f"Episode: {step_info.get('episode_id', 0)} | Step: {step_info.get('step', 0)}",
                f"Task Type: {step_info.get('task_type', 'Unknown')}",
                f"Instruction: {step_info.get('instruction', 'N/A')}",
                f"Arm State: {step_info.get('arm_state', 'Unknown')}",
                f"Action: {step_info.get('actions', 'N/A')}",
                f"Reward: {step_info.get('reward', 0):.3f} | Success: {step_info.get('success', 0):.1f}",
            ]
            
            for line in info_lines:
                draw.text((padding, current_y), line, fill='black', font=normal_font)
                current_y += 18
        
        # User prompt section
        current_y += 20
        if user_prompt:
            draw.text((padding, current_y), "User Prompt (Input to GPT):", fill='purple', font=title_font)
            current_y += 25
            
            # Show truncated prompt - increase visibility
            prompt_text = user_prompt[:800] + "..." if len(user_prompt) > 800 else user_prompt
            wrapped_lines = self._wrap_text(prompt_text, canvas_width - padding * 2, draw, small_font)
            
            for line in wrapped_lines[:12]:  # Show more lines
                draw.text((padding, current_y), line, fill='darkblue', font=small_font)
                current_y += 12
        else:
            # Debug: show if prompt is missing
            draw.text((padding, current_y), "User Prompt: [MISSING]", fill='red', font=title_font)
            current_y += 25
        
        # GPT response section
        current_y += 20
        if gpt_response:
            draw.text((padding, current_y), "GPT-4o-mini Raw Response:", fill='darkgreen', font=title_font)
            current_y += 25
            
            # Show full response - increase visibility
            response_text = gpt_response[:1000] + "..." if len(gpt_response) > 1000 else gpt_response
            wrapped_lines = self._wrap_text(response_text, canvas_width - padding * 2, draw, small_font)
            
            for line in wrapped_lines[:15]:  # Show more lines
                draw.text((padding, current_y), line, fill='darkgreen', font=small_font)
                current_y += 12
        else:
            # Debug: show if response is missing
            draw.text((padding, current_y), "GPT Response: [MISSING]", fill='red', font=title_font)
            current_y += 25
        
        print(f"[IMAGE DEBUG] Final canvas size saved: {canvas.size}")
        return canvas
    
    def _wrap_text(self, text: str, max_width: int, draw, font) -> List[str]:
        """Wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] > max_width and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def _save_results(self, results: Dict):
        """Save episode results to JSON."""
        with open(self.episode_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary plot
        self._create_summary_plot(results)
    
    def _create_summary_plot(self, results: Dict):
        """Create a summary plot for the episode."""
        if not results["steps"]:
            return
        
        steps = [s["step"] for s in results["steps"]]
        rewards = [s["reward"] for s in results["steps"]]
        total_rewards = [s["total_reward"] for s in results["steps"]]
        success = [s["success"] for s in results["steps"]]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        # Rewards
        ax1.plot(steps, rewards, 'b-', alpha=0.6, label='Step Reward')
        ax1.plot(steps, total_rewards, 'g-', label='Cumulative')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax1.set_ylabel('Reward')
        ax1.set_title(f'Episode {self.episode_id}: {results["instruction"][:50]}...')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Success metric
        ax2.plot(steps, success, 'r-', label='Task Success')
        ax2.axhline(y=1.0, color='g', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Success')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.episode_dir / "summary.png", dpi=100)
        plt.close()
    
    def _create_failed_result(self, error: str) -> Dict[str, Any]:
        """Create a result dict for failed episode."""
        return {
            "task_type": self.task_type,
            "episode_id": self.episode_id,
            "instruction": "Failed to start",
            "steps": [],
            "total_reward": 0.0,
            "success": False,
            "error": error,
            "final_metrics": {"success": False}
        }

# ============================================
# Main Test Runner
# ============================================

def run_tests():
    """Run all test episodes."""
    setup_environment()
    logger = setup_logging(TEST_CONFIG["debug_mode"])
    
    logger.info("="*60)
    logger.info("SPOC Testing with GPT-4o-mini")
    logger.info("="*60)
    logger.info(f"Task types: {TEST_CONFIG['task_types']}")
    logger.info(f"Episodes per task: {TEST_CONFIG['episodes_per_task']}")
    logger.info(f"Max steps: {TEST_CONFIG['max_steps']}")
    
    # Initialize model
    model_config = OpenAIModelConfig(
        api_key=API_KEY,
        model_name=MODEL_CONFIG["model_name"],
        max_tokens=MODEL_CONFIG["max_tokens"],
        temperature=MODEL_CONFIG["temperature"],
        seed=MODEL_CONFIG["seed"]
    )
    model = OpenAIModelInterface(model_config)
    
    # Run tests for each task type
    all_results = []
    
    for task_type in TEST_CONFIG["task_types"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {task_type}")
        logger.info(f"{'='*60}")
        
        # Initialize environment for this task type
        env_config = SpocEnvConfig(
            data_path=SPOC_DATA_PATH,
            task_type=task_type,
            chores_split="val",
            prompt_format="grounding_worldmodeling",  # Best SPOC format
            max_actions_per_step=3,
            action_sep=",",
            image_placeholder="<image>"
        )
        
        try:
            env = SpocEnv(env_config)
            
            # Run episodes
            for episode_id in range(TEST_CONFIG["episodes_per_task"]):
                runner = SPOCEpisodeRunner(env, model, task_type, episode_id, logger)
                result = runner.run()
                all_results.append(result)
                
                # Save intermediate results
                save_results_summary(all_results)
            
            env.close()
            
        except Exception as e:
            logger.error(f"Failed to test {task_type}: {e}")
    
    # Final summary
    print_final_summary(all_results)
    save_results_summary(all_results, final=True)
    
    logger.info(f"\n✅ Testing complete! Results saved to: {OUTPUT_DIR}")

def save_results_summary(results: List[Dict], final: bool = False):
    """Save summary of all results."""
    filename = "final_summary.json" if final else "summary.json"
    
    # Calculate statistics
    total = len(results)
    successful = sum(1 for r in results if r.get("success", False))
    
    # Group by task
    by_task = {}
    for r in results:
        task = r["task_type"]
        if task not in by_task:
            by_task[task] = {"total": 0, "success": 0, "rewards": [], "steps": []}
        by_task[task]["total"] += 1
        if r.get("success", False):
            by_task[task]["success"] += 1
        by_task[task]["rewards"].append(r.get("total_reward", 0))
        by_task[task]["steps"].append(r.get("total_steps", 0))
    
    # Calculate averages
    for task in by_task:
        stats = by_task[task]
        stats["success_rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        stats["avg_reward"] = np.mean(stats["rewards"]) if stats["rewards"] else 0
        stats["avg_steps"] = np.mean(stats["steps"]) if stats["steps"] else 0
    
    summary = {
        "total_episodes": total,
        "successful_episodes": successful,
        "overall_success_rate": successful / total if total > 0 else 0,
        "by_task": by_task,
        "detailed_results": results
    }
    
    with open(OUTPUT_DIR / filename, 'w') as f:
        json.dump(summary, f, indent=2)

def print_final_summary(results: List[Dict]):
    """Print final summary to console."""
    total = len(results)
    successful = sum(1 for r in results if r.get("success", False))
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Total Episodes: {total}")
    print(f"Successful: {successful}")
    print(f"Success Rate: {successful/total:.1%}" if total > 0 else "0%")
    
    # By task type
    by_task = {}
    for r in results:
        task = r["task_type"]
        if task not in by_task:
            by_task[task] = {"total": 0, "success": 0}
        by_task[task]["total"] += 1
        if r.get("success", False):
            by_task[task]["success"] += 1
    
    print("\nBy Task Type:")
    for task, stats in by_task.items():
        rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {task}: {rate:.1%} ({stats['success']}/{stats['total']})")
    
    print("="*60)

if __name__ == "__main__":
    run_tests()