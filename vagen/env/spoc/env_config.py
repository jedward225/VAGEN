from dataclasses import dataclass, field
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
    resolution: int = 224
    fov: int = 90
    step_length: float = 0.2  # Base movement step size
    success_threshold: float = 1.0  # Distance threshold for navigation-based success
    multiview: bool = False
    
    # === Prompt and Action Configuration ===
    prompt_format: str = "grounding_worldmodeling"
    max_actions_per_step: int = 1
    action_sep: str = ','
    image_placeholder: str = "<image>"
    special_token_list: List[str] = field(default_factory=lambda: ["<pad>", "<s>", "</s>", "<unk>", "<mask>"])
    
    # === Reward Configuration ===
    format_reward: float = 1.0
    use_state_reward: bool = False
    max_objects_in_state: int = 10

    def config_id(self) -> str:
        """Generate a unique identifier for this configuration."""
        id_fields = ["chores_split", "task_type", "render_mode", "max_actions_per_step"]
        id_str = ",".join([f"{field.name}={getattr(self, field.name)}" for field in fields(self) if field.name in id_fields])
        return f"SpocEnvConfig({id_str})"

if __name__ == "__main__":
    config = SpocEnvConfig()
    print(config.config_id())