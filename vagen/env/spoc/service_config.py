from vagen.env.base.base_service_config import BaseServiceConfig
from dataclasses import dataclass, fields,field

@dataclass
class SpocServiceConfig(BaseServiceConfig):
    devices: list = field(default_factory=lambda: [0, 1, 2, 3])
    use_state_reward: bool = False