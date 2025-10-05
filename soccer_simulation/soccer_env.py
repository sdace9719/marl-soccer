import gymnasium
import numpy as np
import json
import os
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
#from pettingzoo.utils.wrappers import OrderEnforcingWrapper
from typing import Dict, Any, Optional
from game.game import Game

# Match legacy server frame skipping: number of physics ticks per env step
FRAME_SKIPS = 6

# This is the client for a single environment instance.
# It will be parallelized by our custom VecEnv.
class SoccerEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "soccer_sim_v1"}

    def __init__(self, render_mode: Optional[str] = None, config: Optional[Dict[str, Any]] = None, **kwargs):
        # Enforce single-environment usage; ignore allowed values, reject others
        if "env" in kwargs and kwargs["env"] != 1:
            raise ValueError("SoccerEnv supports only a single environment (env must be 1).")
        if "num_envs" in kwargs and kwargs["num_envs"] != 1:
            raise ValueError("SoccerEnv supports only a single environment (num_envs must be 1).")

        self.render_mode = render_mode
        
        # --- PettingZoo API Requirements ---
        self.possible_agents = [f"agent_{i}" for i in range(4)]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = {agent_id: idx for idx, agent_id in enumerate(self.possible_agents)}

        # Normalized action space for RL algorithms (e.g., PPO expects [-1, 1])
        self._action_space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation stacking
        self._stack_size = 3
        # Per-frame features after unit-directional encoding: 4 (vx,vy,angle,ang_vel) + 6*3 = 22
        self._frame_size = 22

        # Load config if not provided
        if config is None:
            config_path_candidates = [
                os.path.join(os.path.dirname(__file__), "config.json"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
            ]
            cfg = None
            for path in config_path_candidates:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        cfg = json.load(f)
                    break
            if cfg is None:
                raise FileNotFoundError("Could not find config.json next to soccer_env.py; pass config explicitly.")
            config = cfg

        # Single embedded game instance
        # Headless unless explicitly rendering via render_mode="human"
        self._game = Game(config=config, headless=(self.render_mode != "human"))

        # Physical action scale (force_x, force_y, torque). Keep backward-compatible defaults.
        physics_cfg = config.get("physics", {}) if isinstance(config, dict) else {}
        self._force_max = float(physics_cfg.get("action_force_max", 150000.0))
        self._torque_max = float(physics_cfg.get("action_torque_max", 100000.0))

        # Unbounded observation space (flattened 3 x 21)
        self._observation_space = Box(low=-np.inf, high=np.inf, shape=(self._frame_size * self._stack_size,), dtype=np.float32)

        # Per-agent observation history buffers
        self._obs_buffers: Dict[str, list] = {agent: [] for agent in self.possible_agents}

        # Renderer is optional and created lazily on first render
        self._renderer = None

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        # Options may contain: {"use_fixed_positions": bool, "use_full_random_positions": bool}
        use_fixed = False
        use_full_random = False
        if isinstance(options, dict):
            use_fixed = bool(options.get("use_fixed_positions", False))
            use_full_random = bool(options.get("use_full_random_positions", False))
        obs_list = self._game.reset(use_fixed_positions=use_fixed, use_full_random_positions=use_full_random, seed=seed)
        # Initialize buffers with 3 identical frames
        observations = {}
        for agent_id, obs in zip(self.possible_agents, obs_list):
            obs_arr = np.asarray(obs, dtype=np.float32)
            self._obs_buffers[agent_id] = [obs_arr.copy(), obs_arr.copy(), obs_arr.copy()]
            stacked = np.concatenate(self._obs_buffers[agent_id], axis=0)
            observations[agent_id] = stacked.astype(np.float32)
        infos = {agent_id: {} for agent_id in self.possible_agents}
        return observations, infos

    def step(self, actions):
        # Ensure all expected agents provide actions
        expected_agents = list(self.possible_agents)
        missing_agents = [aid for aid in expected_agents if aid not in actions]
        if missing_agents:
            raise ValueError(f"Missing actions for agents: {missing_agents}. Expected actions for {expected_agents}.")
        extra_agents = [aid for aid in actions.keys() if aid not in expected_agents]
        if extra_agents:
            raise ValueError(f"Received actions for unknown agents: {extra_agents}. Expected only {expected_agents}.")

        full_actions = {}
        for agent_id in expected_agents:
            act = actions.get(agent_id)
            act_arr = np.asarray(act, dtype=np.float32)
            if act_arr.shape != (3,):
                raise ValueError(f"Action for agent '{agent_id}' must have shape (3,), got {act_arr.shape}.")
            if not np.all(np.isfinite(act_arr)):
                raise ValueError(f"Action contains non-finite values for agent '{agent_id}': {act_arr.tolist()}")
            # Clip to [-1, 1] then scale to physical units
            act_clipped = np.clip(act_arr, -1.0, 1.0)
            scaled = np.array([
                act_clipped[0] * self._force_max,
                act_clipped[1] * self._force_max,
                act_clipped[2] * self._torque_max,
            ], dtype=np.float32)
            full_actions[agent_id] = scaled.tolist()

        # Apply all forces once and advance physics by a single fixed timestep
        obs_list, rewards_list, done, info = self._game.step(full_actions)

        observations = {}
        for agent_id, obs in zip(self.possible_agents, obs_list):
            obs_arr = np.asarray(obs, dtype=np.float32)
            buf = self._obs_buffers.get(agent_id, [])
            if len(buf) == 0:
                buf = [obs_arr.copy(), obs_arr.copy(), obs_arr.copy()]
            else:
                buf = buf[1:] + [obs_arr]
            self._obs_buffers[agent_id] = buf
            stacked = np.concatenate(buf, axis=0)
            observations[agent_id] = stacked.astype(np.float32)
        rewards = {
            "agent_0": float(rewards_list[0]),
            "agent_1": float(rewards_list[1]),
            "agent_2": 0.0,
            "agent_3": 0.0,
        }
        terminations = {agent_id: False for agent_id in self.possible_agents}
        truncations = {agent_id: bool(done) for agent_id in self.possible_agents}
        infos = {agent_id: dict(info) for agent_id in self.possible_agents}

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode != "human":
            return
        if self._renderer is None:
            from renderer import PygameRenderer
            self._renderer = PygameRenderer()
        self._renderer.draw(self._game)

    def close(self):
        try:
            if self._renderer is not None:
                import pygame
                pygame.quit()
                self._renderer = None
        except Exception:
            pass


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



def make_env(**kwargs):
    """
    A simple function to instantiate and return the soccer environment.
    """
    env = soccerenv(**kwargs)
    
    return env


def get_observation_scalers(env: SoccerEnv):
    """
    Returns a dict of maximum ranges used to scale observation components.
    Keys:
      - max_velocity
      - max_angular_velocity
      - field_diagonal
      - stack_size
      - frame_size
    """
    physics_cfg = env._game.config.get("physics", {})
    max_velocity = float(physics_cfg.get("max_velocity", 400.0))
    max_ang_vel = float(physics_cfg.get("max_angular_velocity", physics_cfg.get("action_torque_max", 100000.0) / 100.0))
    from game.constants import SCREEN_WIDTH, SCREEN_HEIGHT
    field_diag = float((SCREEN_WIDTH**2 + SCREEN_HEIGHT**2) ** 0.5)
    return {
        "max_velocity": max_velocity,
        "max_angular_velocity": max_ang_vel,
        "field_diagonal": field_diag,
        "stack_size": env._stack_size,
        "frame_size": env._frame_size,
    }