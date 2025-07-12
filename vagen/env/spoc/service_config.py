from dataclasses import dataclass, field
from vagen.env.base.base_service_config import BaseServiceConfig
from typing import List

@dataclass
class SpocServiceConfig(BaseServiceConfig):
    """
    Configuration specific to the SPOC environment service.
    """
    # Overrides the base environment name
    env_name: str = "spoc"
    # Defines the maximum number of concurrent SPOC environments
    max_workers: int = 48
    # Lists the GPU devices to be used for the SPOC environments
    devices: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    # Whether to use state reward functionality
    use_state_reward: bool = False