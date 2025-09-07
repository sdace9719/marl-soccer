import requests
import numpy as np
from soccer_env import soccer_raw_env as make_pettingzoo_env
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces


class _ActionSpaceTuple:
    def __init__(self, template_env):
        self._env = template_env
        self._agents = template_env.possible_agents

    def sample(self):
        return tuple(self._env.action_space(agent).sample() for agent in self._agents)


class SoccerVecEnv:
    def __init__(self, num_envs, **kwargs):
        if num_envs not in [1, 8]:
            raise ValueError("The number of environments must be 1 or 8.")

        template_env = make_pettingzoo_env(**kwargs)
        self.url = template_env.url
        self.possible_agents = template_env.possible_agents
        self.num_envs = num_envs
        self._action_space = _ActionSpaceTuple(template_env)

    @property
    def action_space(self):
        return self._action_space

    def _flatten_observations(self, observations_per_env):
        # observations_per_env: list of [obs_agent0, obs_agent1, ...] for each env
        return tuple(
            np.array(agent_obs, dtype=np.float32)
            for agent_obs in zip(*observations_per_env)
        )

    def reset(self):
        res = requests.post(f"{self.url}/reset_all")
        res.raise_for_status()
        data = res.json()
        observations = data["observations"]
        return self._flatten_observations(observations)

    def step(self, actions):
        # actions is a tuple of arrays, one per agent, each shaped (num_envs, action_dim)
        actions_list = []
        for i in range(self.num_envs):
            env_actions = {}
            for j, agent in enumerate(self.possible_agents):
                env_actions[agent] = np.asarray(actions[j][i]).tolist()
            actions_list.append(env_actions)

        res = requests.post(f"{self.url}/step", json={"actions": actions_list})
        res.raise_for_status()
        data = res.json()

        step_observations = self._flatten_observations(data["observations"])
        rewards = np.array(data["rewards"], dtype=np.float32)
        dones = np.array(data["dones"], dtype=bool)
        infos = data["infos"]

        # Auto-reset any finished envs
        if np.any(dones):
            reset_res = requests.post(f"{self.url}/reset_all")
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            observations = self._flatten_observations(reset_data["observations"])
        else:
            observations = step_observations

        # For compatibility with tests: duplicate dones into both terminations and truncations
        terminations = dones.copy()
        truncations = dones.copy()
        return observations, rewards, terminations, truncations, infos

    def close(self):
        return None


def vectorized_env(num_envs=8, **kwargs):
    return SoccerVecEnv(num_envs=num_envs, **kwargs)


class RemoteMARLVecEnvSB3(VecEnv):
    def __init__(self, num_envs=8, **kwargs):
        if num_envs not in [1, 8]:
            raise ValueError("The number of environments must be 1 or 8.")

        template_env = make_pettingzoo_env(**kwargs)
        self.url = template_env.url
        self.possible_agents = template_env.possible_agents
        self.num_envs = num_envs

        # Identify trainable (blue team) vs opponent (red team) agents
        # Assumption based on environment: agent_0, agent_1 are blue; agent_2, agent_3 are red
        self.trainable_indices = [0, 1]
        self.opponent_indices = [2, 3]
        self.trainable_agents = [self.possible_agents[i] for i in self.trainable_indices]
        self.opponent_agents = [self.possible_agents[i] for i in self.opponent_indices]

        # Build flattened Box spaces ONLY for trainable agents
        obs_spaces_train = [template_env.observation_space(agent) for agent in self.trainable_agents]
        act_spaces_train = [template_env.action_space(agent) for agent in self.trainable_agents]
        # Keep opponent spaces for random action sampling
        self._opponent_act_spaces = [template_env.action_space(agent) for agent in self.opponent_agents]

        obs_low = np.concatenate([space.low.reshape(-1) for space in obs_spaces_train]).astype(np.float32)
        obs_high = np.concatenate([space.high.reshape(-1) for space in obs_spaces_train]).astype(np.float32)
        act_low = np.concatenate([space.low.reshape(-1) for space in act_spaces_train]).astype(np.float32)
        act_high = np.concatenate([space.high.reshape(-1) for space in act_spaces_train]).astype(np.float32)

        observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        super().__init__(num_envs=self.num_envs, observation_space=observation_space, action_space=action_space)

        # Cache action splits for fast slicing (trainable only)
        self._act_dims = [int(np.prod(space.shape)) for space in act_spaces_train]
        self._act_splits = np.cumsum([0] + self._act_dims)

        self._last_obs = None
        self._waiting_step = False
        self._cached_results = None

    def reset(self):
        res = requests.post(f"{self.url}/reset_all")
        res.raise_for_status()
        data = res.json()
        observations_per_env = data["observations"]  # List[List[obs_per_agent]] length num_envs
        # Flatten per-env observations into single vector per env (trainable agents only)
        flat_obs = []
        for obs_list in observations_per_env:
            obs_list_train = [obs_list[i] for i in self.trainable_indices]
            flat_obs.append(np.concatenate([np.asarray(o, dtype=np.float32).reshape(-1) for o in obs_list_train], axis=0))
        self._last_obs = np.stack(flat_obs, axis=0)
        return self._last_obs

    def step_async(self, actions):
        # Accept actions as np.ndarray shape (num_envs, total_action_dim for trainable agents)
        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions.reshape(self.num_envs, -1)
        actions_list = []
        for i in range(self.num_envs):
            env_actions = {}
            # Learned actions for trainable agents
            for agent_idx, agent in enumerate(self.trainable_agents):
                a = actions[i, self._act_splits[agent_idx]:self._act_splits[agent_idx+1]]
                env_actions[agent] = a.tolist()
            # Random actions for opponent agents
            for opp_idx, agent in enumerate(self.opponent_agents):
                env_actions[agent] = self._opponent_act_spaces[opp_idx].sample().tolist()
            actions_list.append(env_actions)

        res = requests.post(f"{self.url}/step", json={"actions": actions_list})
        res.raise_for_status()
        self._cached_results = res.json()
        self._waiting_step = True

    def step_wait(self):
        assert self._waiting_step, "step_wait() called without step_async()"
        data = self._cached_results
        self._cached_results = None
        self._waiting_step = False

        observations_per_env = data["observations"]
        flat_obs = []
        for obs_list in observations_per_env:
            obs_list_train = [obs_list[i] for i in self.trainable_indices]
            flat_obs.append(np.concatenate([np.asarray(o, dtype=np.float32).reshape(-1) for o in obs_list_train], axis=0))
        observations = np.stack(flat_obs, axis=0)

        # Reduce multi-agent rewards to single scalar per env (sum)
        rewards_ma = np.array(data["rewards"], dtype=np.float32)  # shape (num_envs, num_blue_agents)
        rewards = rewards_ma.sum(axis=1)

        dones = np.array(data["dones"], dtype=bool)
        # Ensure infos is a list[dict]
        infos = data["infos"]
        if not isinstance(infos, list):
            infos = [{} for _ in range(self.num_envs)]
        else:
            infos = [info if isinstance(info, dict) else {} for info in infos]

        # Before auto-reset, save final observations for done envs
        if np.any(dones):
            final_observations = observations.copy()
            for env_idx, done in enumerate(dones):
                if done:
                    infos[env_idx]["final_observation"] = final_observations[env_idx]

            # Auto-reset and supply fresh observations for done envs
            reset_res = requests.post(f"{self.url}/reset_all")
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            reset_flat = []
            for obs_list in reset_data["observations"]:
                obs_list_train = [obs_list[i] for i in self.trainable_indices]
                reset_flat.append(np.concatenate([np.asarray(o, dtype=np.float32).reshape(-1) for o in obs_list_train], axis=0))
            reset_obs = np.stack(reset_flat, axis=0)
            observations[dones] = reset_obs[dones]

        # SB3 VecEnv expects 4-tuple with combined dones
        return observations, rewards, dones, infos

    def close(self):
        return None

    # Optional helpers for SB3 API completeness
    def get_attr(self, attr_name, inds=None):
        if attr_name == "possible_agents":
            return self.possible_agents
        if attr_name == "url":
            return self.url
        raise AttributeError(attr_name)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError

    def set_attr(self, attr_name, value, inds=None):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        # No local wrappers are applied; report False for all envs
        if indices is None:
            return [False] * self.num_envs
        if isinstance(indices, (list, tuple, np.ndarray)):
            return [False] * len(indices)
        return [False]


def sb3_vectorized_env(num_envs=8, **kwargs) -> VecEnv:
    return RemoteMARLVecEnvSB3(num_envs=num_envs, **kwargs)


if __name__ == "__main__":
    # --- Example Usage ---
    # This creates a single game instance and wraps it for SB3
    vec_env = vectorized_env()
    
    print("Starting vectorized environment loop...")
    
    # Note: reset() for SB3-style vec envs does not return info
    observations = vec_env.reset()
    
    for _ in range(100): # Run for 100 steps
        # The vectorized environment expects a batch of actions (one for each agent)
        actions = vec_env.action_space.sample()
        
        # Step the environment
        observations, rewards, terminations, truncations, infos = vec_env.step(actions)
        
        # Check if any agent is done
        if any(terminations) or any(truncations):
            print("An episode finished. Vec env will auto-reset.")

    print("Vectorized environment loop finished.")
    vec_env.close()
