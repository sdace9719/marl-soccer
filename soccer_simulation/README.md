## Soccer Simulation for Multi-Agent Reinforcement Learning

This project provides a 2v2 soccer simulation for multi-agent reinforcement learning (MARL). It uses a client–server architecture where clients send actions and receive observations via a simple HTTP API. You can interact with it using:
- A PettingZoo-compatible single-environment client (`soccerenv`)
- A custom vectorized client for 1 or 8 parallel environments (for testing)
- An SB3-compatible `VecEnv` for Stable-Baselines3 training

## Architecture

- Server: Flask-based API with two endpoints:
  - `POST /reset_all` → returns initial observations for all environments (1 or 8)
  - `POST /step` → accepts a batched action payload and returns observations, rewards, done flags, and infos per env
- Physics/game loop: Runs in the server process; configurable via `config.json`.
- Clients: Lightweight Python clients in this repo batch actions and talk to the API.

### Configuration
- File: `config.json`
- Tunables:
  - Physics: `max_velocity`, `agent_mass`, `ball_mass`, `agent_friction`, `ball_friction`
  - Rewards: shaping and terminal rewards
  - Simulation: `max_steps` (episode length)
- To change values, edit `config.json` and restart the server.

## File Structure
- `run.py`: Starts the server. Supports `--headless` and `--num-envs {1,8}`.
- `soccer_env.py`:
  - `SoccerEnv`: PettingZoo ParallelEnv client for a single environment
  - `soccer_raw_env()`: returns raw `SoccerEnv`
  - `soccerenv()`: returns wrapped env (alias used in tests/imports)
- `vectorized_environment.py`:
  - `vectorized_env(num_envs)`: simple vectorized client (1 or 8)
  - `sb3_vectorized_env(num_envs)`: SB3 `VecEnv` that flattens obs/actions for training
- `test_environments.py`: Examples for single PettingZoo and vectorized usage
- `test_api.py`: Minimal API test driving the server directly
- `soccer_simulation/api/`: Flask server and environment manager
- `soccer_simulation/game/`: Game logic and entities

## Getting Started

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Start the Server
```bash
python run.py --headless --num-envs 8   # or --num-envs 1
```

### 3) Single-Env (PettingZoo) Client
```python
from soccer_env import soccerenv

env = soccerenv()  # single env only
obs, infos = env.reset()

done = False
while not done:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    done = all(truncations.values())  # episodes end by step count
```

### 4) Vectorized Client (Testing: 1 or 8 envs)
```python
import numpy as np
from vectorized_environment import vectorized_env

num_envs = 8
vec = vectorized_env(num_envs=num_envs)
obs = vec.reset()

for _ in range(1000):
    actions = tuple(np.stack(x) for x in zip(*[vec.action_space.sample() for _ in range(num_envs)]))
    obs, rewards, terminations, truncations, infos = vec.step(actions)
    # rewards shape: (num_envs, 2) => per-env, per-agent
    if truncations.all():
        obs = vec.reset()
```

### 5) Stable-Baselines3 (SB3) Training
```python
from vectorized_environment import sb3_vectorized_env
from stable_baselines3 import PPO

vec = sb3_vectorized_env(num_envs=8)
model = PPO("MlpPolicy", vec, verbose=1)
model.learn(total_timesteps=1000000)
```

Notes:
- SB3 `VecEnv` flattens per-agent observations into a single vector per env and sums per-env rewards into a scalar (suitable for single-agent algorithms controlling a team).
- If you need per-agent reward logging with SB3, extend `infos[env_idx]['per_agent_rewards']` in `sb3_vectorized_env` and record it via callbacks.

## Testing
- API smoke test:
```bash
python test_api.py --num-envs 8
```
- Client tests:
```bash
python -c "from test_environments import test_pettingzoo_env; test_pettingzoo_env(3)"
python -c "from test_environments import test_vectorized_env; test_vectorized_env(num_envs=8, total_episodes_to_finish=10)"
```

## Tips & Conventions
- The server supports exactly 1 or 8 environments; clients enforce this.
- `SoccerEnv` is single-env only and will raise if initialized with other counts.
- Vectorized rewards are returned as `(num_envs, 2)` for the two controlled agents.
- Goal detection:
  - Single env: `infos['agent_0'].get('goal_scored_by')`
  - Vectorized: iterate `for env_idx, info in enumerate(infos): info.get('goal_scored_by')`

## Changing Physics/Rewards
- Edit `config.json` and restart the server.
- Example keys:
  - `physics.max_velocity`, `physics.agent_friction`, `physics.ball_friction`
  - `rewards.goal_scored_reward`, `rewards.alive_penalty`
  - `simulation.max_steps`
