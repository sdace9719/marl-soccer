## Soccer Simulation for Multi-Agent Reinforcement Learning

This project provides a 2v2 soccer simulation for multi-agent reinforcement learning (MARL). It now ships a self-contained PettingZoo `ParallelEnv`. You can interact with it using:
- A PettingZoo-compatible single-environment client (`soccerenv`) — self-contained

## Architecture

- Embedded single-env: `soccer_simulation/soccer_env.py` implements a PettingZoo `ParallelEnv` that directly uses the internal `Game` for simulation.
- Physics/game loop: Implemented in `soccer_simulation/game/`. Tuned via `config.json`.

### Configuration
- File: `config.json`
- Tunables:
  - Physics: `max_velocity`, `agent_mass`, `ball_mass`, `agent_friction`, `ball_friction`
  - Rewards: shaping and terminal rewards
  - Simulation: `max_steps` (episode length)
- To change values, edit `config.json` and restart the server.

## File Structure
- `soccer_env.py`:
  - `SoccerEnv`: PettingZoo `ParallelEnv` (self-contained, single env)
  - `soccer_raw_env()`: returns raw `SoccerEnv`
  - `soccerenv()`: returns wrapped env (alias used in tests/imports)
- `renderer.py`: Optional Pygame renderer used by `SoccerEnv.render()` when `render_mode="human"`
- `soccer_simulation/game/`: Game logic and entities
- `pz_api_lint.py`: PettingZoo Parallel API linter runner for compliance checks

## Getting Started

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Single-Env (PettingZoo) Client — self-contained
```python
from soccer_simulation.soccer_env import soccerenv

env = soccerenv(render_mode=None)  # or render_mode="human" to view
obs, infos = env.reset()

done = False
while not done:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)
    done = all(truncations.values())
env.close()
```

## Testing
- PettingZoo API linter:
```bash
python -m pz_api_lint
```
- Reward tests (self-contained env; no server needed):
```bash
python -m test_rewards
```

## Tips & Conventions
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
 
## Result

![soccer](soccer_simulation/soccer-twos.gif?raw=true)

## Future work
It can be seen that the opponent is not challenging enough so it easy for our trained model to score. For SOTA performance, proper curriculum based training has to be implemented. To even achieve this, I had to put significant work and debug a lot(typical of any rl training).

![Alt text](soccer_simulation/Honest_Work.jpg?raw=true)
