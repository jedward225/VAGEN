from vagen.env.spoc.env_config import SpocEnvConfig
from vagen.env.spoc.env import SpocEnv

env = SpocEnv(SpocEnvConfig(gpu_device=0))
print("SpocEnv init OK!")
env.close()