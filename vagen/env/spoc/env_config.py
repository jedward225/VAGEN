from dataclasses import dataclass, field, fields
from vagen.env.base.base_env_config import BaseEnvConfig
from typing import List, Optional

@dataclass
class SpocEnvConfig(BaseEnvConfig):
    """Configuration for the SPOC environment."""
    
    # === Dataset Configuration ===
    # Path to the root of the SPOC dataset (e.g., '/path/to/fifteen_type')
    data_path: str = "/path/to/your/spoc/dataset/fifteen_type" 
    # Task type to load from the dataset (e.g., "FetchType", "ObjectNavType")
    task_type: str = "FetchType"
    # Dataset split to use
    chores_split: str = "train" 

    # === Environment Configuration ===
    env_name: str = "spoc"
    render_mode: str = "vision"  # "vision" or "text" 
    resolution: int = 224
    fov: int = 90
    step_length: float = 0.2  # Base movement step size
    success_threshold: float = 1.0  # Distance threshold for navigation-based success
    multiview: bool = False
    
    # === Top-Down Map Configuration ===
    include_top_down_map: bool = False  # Whether to include top-down map in observations
    map_size: int = 256  # Size of the top-down map (map_size x map_size)
    include_agent_path: bool = False  # Whether to visualize agent's path on the map
    path_width: float = 0.045  # Width of the path visualization
    
    # === Prompt and Action Configuration ===
    prompt_format: str = "grounding_worldmodeling"
    max_actions_per_step: int = 3  # Increased from 1 to allow multi-step coordination
    action_sep: str = ','
    image_placeholder: str = "<image>"
    special_token_list: List[str] = field(default_factory=lambda: ["<pad>", "<s>", "</s>", "<unk>", "<mask>"])
    
    # === Reward Configuration ===
    format_reward: float = 1.0  # Will be reduced to 0.1 in new reward system
    use_state_reward: bool = False
    max_objects_in_state: int = 10
    
    # === GPU Configuration ===
    gpu_device: int = 0

    def config_id(self) -> str:
        """Generate a unique identifier for this configuration."""
        id_fields = ["chores_split", "task_type", "render_mode", "max_actions_per_step"]
        id_str = ",".join([f"{field.name}={getattr(self, field.name)}" for field in fields(self) if field.name in id_fields])
        return f"SpocEnvConfig({id_str})"

if __name__ == "__main__":
    config = SpocEnvConfig()
    print(config.config_id())