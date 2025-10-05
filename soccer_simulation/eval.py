import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from soccer_env import soccerenv # Make sure soccer_env.py is in the same directory
from torch.distributions.normal import Normal

# NOTE: Agent class and layer_init are assumed to be defined as in your script.
# (Included here for completeness)
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, device='cpu'):
        super().__init__()
        if hasattr(envs, 'single_observation_space'):
            obs_space = envs.single_observation_space
            act_space = envs.single_action_space
        else:
            any_agent = envs.possible_agents[0]
            obs_space = envs.observation_space(any_agent)
            act_space = envs.action_space(any_agent)
        obs_shape = np.prod(obs_space.shape)
        act_shape = np.prod(act_space.shape)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 512)), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(),
            layer_init(nn.Linear(128, 64)), nn.Tanh(), layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_shape, 512)), nn.Tanh(),
            nn.Linear(512, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(),
            layer_init(nn.Linear(128, 64)), nn.Tanh(), layer_init(nn.Linear(64, act_shape), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_shape))
        self.device=device
    
    def get_deterministic_action(self, x):
        action_mean = self.actor_mean(x)
        return action_mean

# --- 1. SETUP AND LOAD MODEL/NORMALIZER ---

# Define the run name to load from
run_name = "run5" 

device = torch.device("cpu") 

# Create a dummy environment to initialize the Agent class correctly
dummy_env = soccerenv()
agent = Agent(dummy_env, device=device).to(device)
dummy_env.close()

# Load the saved model weights
model_path = f"runs/{run_name}/ppo_pettingzoo_soccer.ppo_model"
agent.load_state_dict(torch.load(model_path, map_location=device))
agent.eval()  # Set the agent to evaluation mode

normalizer_path = f"runs/{run_name}/latest_normalizer_stats.npz"
normalizer_stats = np.load(normalizer_path)
obs_mean = normalizer_stats['mean']
obs_var = normalizer_stats['var']
obs_std = np.sqrt(obs_var)

# --- 2. RUN THE EVALUATION LOOP ---
env = soccerenv(render_mode="human")
num_episodes = 5

trainable_agent_ids = ["agent_0", "agent_1"]
act_shape = env.action_space(trainable_agent_ids[0]).shape

for ep in range(num_episodes):
    obs, infos = env.reset()
    final_score = {"blue": 0, "red": 0}
    # Track average rewards per episode for trainable agents
    episode_steps = 0
    episode_returns = {agent_id: 0.0 for agent_id in trainable_agent_ids}

    while env.agents:
        trainable_obs = np.stack([obs[a] for a in trainable_agent_ids])
        
        # This normalization will now work correctly on a per-feature basis
        normalized_obs = np.clip((trainable_obs - obs_mean) / (obs_std + 1e-8), -10, 10)
        obs_tensor = torch.tensor(normalized_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            raw_actions = agent.get_deterministic_action(obs_tensor).cpu().numpy()
            # action_mean = agent.actor_mean(obs_tensor)
            # action_logstd = agent.actor_logstd.expand_as(action_mean)
            # action_std = torch.exp(action_logstd)
            # probs = Normal(action_mean, action_std)
            # raw_actions = probs.sample().cpu().numpy()
        
        actions = {}
        for i, agent_id in enumerate(trainable_agent_ids):
            actions[agent_id] = raw_actions[i]
        
        for agent_id in env.agents:
            if agent_id not in trainable_agent_ids:
                actions[agent_id] = np.random.uniform(-1.0, 1.0, size=act_shape).astype(np.float32)

        obs, rewards, terminations, truncations, infos = env.step(actions)
        episode_steps += 1
        for agent_id in trainable_agent_ids:
            episode_returns[agent_id] += float(rewards.get(agent_id, 0.0))
        
        env.render()
        #time.sleep(1/60)

        if not env.agents:
            any_agent = next(iter(infos))
            if "score" in infos[any_agent]:
                final_score = infos[any_agent]["score"]
            
    # Log per-episode average rewards (sum over steps) for the trainable agents
    episode_totals = {aid: episode_returns[aid] for aid in trainable_agent_ids}
    team_mean = float(np.mean(list(episode_totals.values())))
    print(f"Episode {ep+1} average reward per episode: {episode_totals}, team_mean={team_mean:.4f}")
    print(f"Episode {ep+1} final score: Blue={final_score.get('blue',0)}, Red={final_score.get('red',0)}")

env.close()