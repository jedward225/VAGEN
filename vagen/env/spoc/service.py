from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from vagen.env.base.base_service import BaseService
from .env import SpocEnv
from .env_config import SpocEnvConfig
from vagen.server.serial import serialize_observation
from .service_config import SpocServiceConfig
from vagen.env.utils.state_reward_text_utils import service_state_reward_wrapper

class SpocService(BaseService):
    """
    Service class for SPOC environments with Stretch robot manipulation tasks.
    Implements batch operations with parallel processing for efficiency.
    """
    
    def __init__(self, config:SpocServiceConfig):
        """
        Initialize the SpocService.
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.max_workers = config.max_workers
        self.device_status={device_id:set() for device_id in config.devices}
        self.environments = {}
        self.env_configs = {}
        self.config=config
        print(f"[DEBUG] {self.config}")
    
    def create_environments_batch(self, ids2configs: Dict[str, Any]) -> None:
        """
        Create multiple Navigation environments in parallel.
        
        Args:
            ids2configs: A dictionary where each key is an environment ID and the corresponding
                        value is the configuration for that environment.
                Each config should contain:
                - env_name: Should be "navigation"
                - env_config: Navigation specific configuration
        """
        # Define worker function
        def create_single_env(env_id, config):
            try:
                # Verify environment type
                env_name = config.get('env_name', 'navigation')
                if env_name not in ('navigation', 'spoc'):
                    return env_id, None, f"Expected environment type 'navigation', got '{env_name}'"
                
                env_config_dict = config['env_config']
                env_config = SpocEnvConfig(**env_config_dict)
                env = SpocEnv(env_config)
                return env_id, (env, env_config), None
            except Exception as e:
                return env_id, None, f"Failed to create environment: {str(e)}"
           
        
        for i, env_id in enumerate(ids2configs.keys()):
            # Select GPU with the least load
            selected_gpu = min(self.device_status, key=lambda x: len(self.device_status[x]))
            ids2configs[env_id]['env_config']['gpu_device'] = selected_gpu
            self.device_status[selected_gpu].add(env_id)
            
        # Use ThreadPoolExecutor for parallel creation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all environment creation tasks
            futures = {
                executor.submit(create_single_env, env_id, config): env_id 
                for env_id, config in ids2configs.items()
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                try:
                    env_id, result, error = future.result()
                    if error:
                        print(f"Error creating environment {env_id}: {error}")
                        # 如果创建失败，从 device_status 中移除该 env_id，防止后续 reset/step 找不到 env
                        for env_set in self.device_status.values():
                            env_set.discard(env_id)
                        continue
                    
                    env, env_config = result
                    self.environments[env_id] = env
                    self.env_configs[env_id] = env_config
                    print(f"Successfully created environment {env_id}")
                except Exception as e:
                    print(f"Exception creating environment {env_id}: {e}")
                    import traceback
                    traceback.print_exc()
    
    def reset_batch(self, ids2seeds: Dict[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """
        Reset multiple Navigation environments in parallel.
        
        Args:
            ids2seeds: A dictionary where each key is an environment ID and the corresponding
                     value is a seed value (or None for using default seeding behavior).
            
        Returns:
            A dictionary mapping environment IDs to tuples of the form (observation, info)
        """
        results = {}
        
        # Define worker function
        def reset_single_env(env_id, seed):     
            if env_id not in self.environments:
                # Return properly formatted error observation
                error_obs = {
                    "obs_str": f"Error: Environment {env_id} not found in service",
                    "multi_modal_data": {}
                }
                error_info = {
                    "error": f"Environment {env_id} not found",
                    "metrics": {
                        "turn_metrics": {"action_is_valid": False, "action_is_effective": False},
                        "traj_metrics": {"success": False}
                    }
                }
                return env_id, (error_obs, error_info), f"Environment {env_id} not found in service"
            
            try:
                env = self.environments[env_id]
                observation, info = env.reset(seed=seed)
                serialized_observation = serialize_observation(observation)
                return env_id, (serialized_observation, info), None
            except Exception as e:
                # Return properly formatted error observation
                error_obs = {
                    "obs_str": f"Error resetting environment {env_id}: {str(e)}",
                    "multi_modal_data": {}
                }
                error_info = {
                    "error": str(e),
                    "metrics": {
                        "turn_metrics": {"action_is_valid": False, "action_is_effective": False},
                        "traj_metrics": {"success": False}
                    }
                }
                return env_id, (error_obs, error_info), str(e)
            
        
        # Use ThreadPoolExecutor for parallel reset
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all reset tasks
            futures = {
                executor.submit(reset_single_env, env_id, seed): env_id 
                for env_id, seed in ids2seeds.items()
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error resetting environment {env_id}: {error}")
                    # Return properly formatted error observation
                    error_obs = {
                        "obs_str": f"Error resetting environment {env_id}: {error}",
                        "multi_modal_data": {}
                    }
                    error_info = {
                        "error": error,
                        "metrics": {
                            "turn_metrics": {"action_is_valid": False, "action_is_effective": False},
                            "traj_metrics": {"success": False}
                        }
                    }
                    results[env_id] = (error_obs, error_info)
                else:
                    results[env_id] = result
        
        return results
    
    @service_state_reward_wrapper
    def step_batch(self, ids2actions: Dict[str, Any]) -> Dict[str, Tuple[Dict, float, bool, Dict]]:
        """
        Take a step in multiple Navigation environments in parallel.
        
        Args:
            ids2actions: A dictionary where each key is an environment ID and the corresponding
                       value is the action to execute in that environment.
            
        Returns:
            A dictionary mapping environment IDs to tuples of the form (observation, reward, done, info)
        """
        results = {}
        
        # Define worker function
        def step_single_env(env_id, action):
            if env_id not in self.environments:
                # Return properly formatted error observation
                error_obs = {
                    "obs_str": f"Error: Environment {env_id} not found in service",
                    "multi_modal_data": {}
                }
                error_info = {
                    "error": f"Environment {env_id} not found",
                    "metrics": {
                        "turn_metrics": {"action_is_valid": False, "action_is_effective": False},
                        "traj_metrics": {"success": False}
                    },
                    "llm_raw_response": action  # 添加llm_raw_response键
                }
                return env_id, (error_obs, 0.0, True, error_info), f"Environment {env_id} not found"
            
            env = self.environments[env_id]
            try:
                observation, reward, done, info = env.step(action)
            except Exception as e:
                print(f"Error stepping navigation environment {env_id}: {e}")
                try:
                    observation, info = env.reset()
                    reward = 0.0
                    done = True 
                except Exception as e:
                    print(f"Error resetting navigation environment {env_id} after step failure: {e}")
                    try:
                        env.close()
                        config = self.env_configs[env_id]
                        env = SpocEnv(config)
                        self.environments[env_id] = env
                        observation, info = env.reset()
                        reward = 0.0
                        done = True
                    except Exception as e2:
                        print(f"Failed to recreate environment {env_id}: {e2}")
                        # Return properly formatted error observation
                        error_obs = {
                            "obs_str": f"Error: Failed to step/reset environment {env_id}: {str(e2)}",
                            "multi_modal_data": {}
                        }
                        error_info = {
                            "error": str(e2),
                            "metrics": {
                                "turn_metrics": {"action_is_valid": False, "action_is_effective": False},
                                "traj_metrics": {"success": False}
                            },
                            "llm_raw_response": action  # 添加llm_raw_response键
                        }
                        return env_id, (error_obs, 0.0, True, error_info), str(e2)
                    
            serialized_observation = serialize_observation(observation)
            return env_id, (serialized_observation, reward, done, info), None
            
        
        # Use ThreadPoolExecutor for parallel step
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all step tasks
            futures = {
                executor.submit(step_single_env, env_id, action): env_id 
                for env_id, action in ids2actions.items()
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error stepping environment {env_id}: {error}")
                    # Return properly formatted error observation
                    error_obs = {
                        "obs_str": f"Error stepping environment {env_id}: {error}",
                        "multi_modal_data": {}
                    }
                    error_info = {
                        "error": error,
                        "metrics": {
                            "turn_metrics": {"action_is_valid": False, "action_is_effective": False},
                            "traj_metrics": {"success": False}
                        },
                        "llm_raw_response": ""  # 这里action不可用，使用空字符串
                    }
                    results[env_id] = (error_obs, 0.0, True, error_info)
                else:
                    results[env_id] = result
        
        return results
    
    def compute_reward_batch(self, env_ids: List[str]) -> Dict[str, float]:
        """
        Compute the total reward for multiple Navigation environments in parallel.
        
        Args:
            env_ids: A list of environment IDs
            
        Returns:
            A dictionary mapping each environment ID to its computed total reward
        """
        results = {}
        
        # Define worker function
        def compute_reward_single_env(env_id):
            if env_id not in self.environments:
                return env_id, 0.0, f"Environment {env_id} not found in service"
            try:
                env = self.environments[env_id]
                return env_id, env.compute_reward(), None
            except Exception as e:
                print(f"Error computing reward for environment {env_id}: {e}")
                return env_id, 0.0, str(e)
           
        
        # Use ThreadPoolExecutor for parallel computation
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all computation tasks
            futures = {
                executor.submit(compute_reward_single_env, env_id): env_id 
                for env_id in env_ids
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error computing reward for environment {env_id}: {error}")
                    results[env_id] = 0.0
                else:
                    results[env_id] = result
        
        return results
    
    def get_system_prompts_batch(self, env_ids: List[str]) -> Dict[str, str]:
        """
        Get system prompts for multiple Navigation environments in parallel.
        
        Args:
            env_ids: A list of environment IDs
            
        Returns:
            A dictionary mapping each environment ID to its corresponding system prompt string
        """
        results = {}
        
        # Define worker function
        def get_system_prompt_single_env(env_id):
            if env_id not in self.environments:
                return env_id, f"Error: Environment {env_id} not found in service", f"Environment {env_id} not found"
            try:
                env = self.environments[env_id]
                return env_id, env.system_prompt(), None
            except Exception as e:
                print(f"Error getting system prompt for environment {env_id}: {e}")
                return env_id, f"Error: {str(e)}", str(e)
       
        
        # Use ThreadPoolExecutor for parallel retrieval
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all retrieval tasks
            futures = {
                executor.submit(get_system_prompt_single_env, env_id): env_id 
                for env_id in env_ids
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                env_id = futures[future]
                env_id, result, error = future.result()
                if error:
                    print(f"Error getting system prompt for environment {env_id}: {error}")
                    results[env_id] = ""
                else:
                    results[env_id] = result
        
        return results
    
    def close_batch(self, env_ids: Optional[List[str]] = None) -> None:
        """
        Close multiple Navigation environments and clean up resources in parallel.
        
        Args:
            env_ids: Optional list of environment IDs to close. If None, close all environments.
        """
        # If no env_ids provided, close all environments
        if env_ids is None:
            env_ids = list(self.environments.keys())
        
        # Define worker function
        def close_single_env(env_id):      
            if env_id not in self.environments:
                print(f"Warning: Environment {env_id} not found during close")
                return None
            try:
                env = self.environments[env_id]
                env.close()
                return None
            except Exception as e:
                print(f"Error closing environment {env_id}: {e}")
                return str(e)
            
        
        # Use ThreadPoolExecutor for parallel closing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all closing tasks
            futures = [executor.submit(close_single_env, env_id) for env_id in env_ids]
            
            # Wait for all tasks to complete
            for future in as_completed(futures):
                error = future.result()
                if error:
                    print(f"Error closing environment: {error}")
        
        # Remove closed environments from dictionaries
        for env_id in env_ids:
            self.environments.pop(env_id, None)
            self.env_configs.pop(env_id, None)
            # 正确地从每块 GPU 的 env 集合中移除该 env_id，而不是把 gpu 键删掉
            for env_set in self.device_status.values():
                env_set.discard(env_id)