import gymnasium
import numpy as np
import requests
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
#from pettingzoo.utils.wrappers import OrderEnforcingWrapper

# This is the client for a single environment instance.
# It will be parallelized by our custom VecEnv.
class SoccerEnv(ParallelEnv):
    metadata = {"render_modes": [], "name": "soccer_sim_v1"}

    def __init__(self, host="localhost", port=5000, render_mode=None, **kwargs):
        # Enforce single-environment usage; ignore allowed values, reject others
        if "env" in kwargs and kwargs["env"] != 1:
            raise ValueError("SoccerEnv supports only a single environment (env must be 1).")
        if "num_envs" in kwargs and kwargs["num_envs"] != 1:
            raise ValueError("SoccerEnv supports only a single environment (num_envs must be 1).")
        self.url = f"http://{host}:{port}"
        self.render_mode = render_mode
        
        # --- PettingZoo API Requirements ---
        self.possible_agents = [f"agent_{i}" for i in range(4)]
        self.agents = self.possible_agents[:]

        self._action_space = Box(
            low=np.array([-150000, -150000, -1e5], dtype=np.float32),
            high=np.array([150000, 150000, 1e5], dtype=np.float32),
            dtype=np.float32,
        )
        
        self._observation_space = Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )
        # We don't interact with the server on init.
        # State will be set by the VecEnv's first reset.
        self.state = None

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed=None, options=None):
        # In vectorized mode, state is provided externally before reset is called.
        # In standalone mode (PettingZoo-only), request a reset from the server.
        self.agents = self.possible_agents[:]
        if self.state is None:
            res = requests.post(f"{self.url}/reset_all")
            res.raise_for_status()
            data = res.json()
            initial_obs_list = data["observations"][0]
            # Set default values for rewards/done/info
            self.state = (initial_obs_list, [0.0, 0.0], False, {})

        observations = {
            agent_id: np.array(obs, dtype=np.float32)
            for agent_id, obs in zip(self.possible_agents, self.state[0])
        }
        infos = {agent_id: {} for agent_id in self.possible_agents}
        return observations, infos

    def step(self, actions):
        # Standalone mode: send actions to server. Vectorized mode passes actions=None
        if actions is not None:
            payload_actions = {
                agent_id: np.asarray(act, dtype=np.float32).tolist()
                for agent_id, act in actions.items()
            }
            res = requests.post(f"{self.url}/step", json={"actions": [payload_actions]})
            res.raise_for_status()
            data = res.json()
            obs_list = data["observations"][0]
            rewards_list = data["rewards"][0]
            done = data["dones"][0]
            info = data["infos"][0]
            # Update state for consistency with vectorized path
            self.state = (obs_list, rewards_list, done, info)
        else:
            obs_list, rewards_list, done, info = self.state

        observations = {
            agent_id: np.array(obs, dtype=np.float32)
            for agent_id, obs in zip(self.possible_agents, obs_list)
        }
        rewards = {
            "agent_0": rewards_list[0],
            "agent_1": rewards_list[1],
            "agent_2": 0.0,
            "agent_3": 0.0,
        }
        terminations = {agent_id: done for agent_id in self.possible_agents}
        truncations = {agent_id: done for agent_id in self.possible_agents}
        infos = {agent_id: info for agent_id in self.possible_agents}

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos


def soccer_raw_env(**kwargs):
    """
    Return the raw, unwrapped environment.
    """
    return SoccerEnv(**kwargs)


def soccerenv(**kwargs):
    """
    Return the wrapped environment with standard wrappers for safety and proper API usage.
    """
    environment = soccer_raw_env(**kwargs)
    #environment = OrderEnforcingWrapper(environment)
    return environment
