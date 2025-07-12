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
    # AI2-THOR 是重量级 Unity 进程，数量过多会导致 CPU 饱和、初始化失败。
    # 对单机服务器而言，4~8 个并行实例更为安全。
    max_workers: int = 4
    # Lists the GPU devices to be used for the SPOC environments
    devices: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    # Whether to use state reward functionality
    use_state_reward: bool = False