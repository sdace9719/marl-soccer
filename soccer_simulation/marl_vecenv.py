import numpy as np

class SyncMultiAgentVecEnv:
    """
    A simple synchronous wrapper for running multiple PettingZoo ParallelEnv instances.
    This respects the multi-agent API and does not flatten agents.
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        
        # These are for API compatibility with libraries that expect single_observation_space
        self.single_observation_space = self.envs[0].observation_space(self.envs[0].possible_agents[0])
        self.single_action_space = self.envs[0].action_space(self.envs[0].possible_agents[0])
        
        self.possible_agents = self.envs[0].possible_agents

    def reset(self, options=None, seed=None):
        """Resets all environments and stacks the observations."""
        all_obs = []
        for i, env in enumerate(self.envs):
            # Seed each environment with a different seed for diversity
            obs, info = env.reset(options=options, seed=seed + i if seed is not None else None)
            # Convert dict of obs to a stacked numpy array
            all_obs.append(self._dict_to_array(obs))
        
        # Stack observations from all envs into a single batch
        return np.stack(all_obs)

    def step(self, actions):
        """
        Steps all environments.
        
        Args:
            actions (np.ndarray): A batch of actions of shape (num_envs, num_agents, action_dim)
        """
        all_obs, all_rews, all_terms, all_truncs, all_infos = [], [], [], [], []
        
        for i, env in enumerate(self.envs):
            # Convert the action array for this env back to a dict
            action_dict = self._array_to_dict(actions[i])
            obs, rew, term, trunc, info = env.step(action_dict)

            # Detect episode end for this env (any agent terminated or truncated)
            done_env = bool(any(term.values()) or any(trunc.values()))

            # By default, return the step observation; if done, auto-reset and return reset obs
            if done_env:
                # Preserve final step info and rewards, but provide next episode's initial observation
                reset_obs, _ = env.reset(options={"use_full_random_positions": True})
                obs_array = self._dict_to_array(reset_obs)
            else:
                obs_array = self._dict_to_array(obs)

            all_obs.append(obs_array)
            all_rews.append(self._dict_to_array(rew))
            all_terms.append(self._dict_to_array(term))
            all_truncs.append(self._dict_to_array(trunc))
            # Keep per-env per-agent infos (dict keyed by agent)
            all_infos.append(info)

        return (
            np.stack(all_obs),
            np.stack(all_rews),
            np.stack(all_terms),
            np.stack(all_truncs),
            all_infos
        )
        
    def _dict_to_array(self, data_dict):
        """Converts a dict of agent data to a numpy array."""
        return np.array([data_dict[agent] for agent in self.possible_agents])

    def _array_to_dict(self, data_array):
        """Converts a numpy array of agent data back to a dict."""
        return {agent: data_array[i] for i, agent in enumerate(self.possible_agents)}

    def close(self):
        for env in self.envs:
            env.close()