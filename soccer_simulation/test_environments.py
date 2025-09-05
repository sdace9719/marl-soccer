from soccer_env import soccerenv
from vectorized_environment import vectorized_env
import numpy as np


def test_pettingzoo_env(num):
    """Tests the standard PettingZoo environment."""
    print("--- Testing PettingZoo Environment ---")
    
    # Create the environment
    env = soccerenv()
    observations, infos = env.reset()
    
    print("Reset successful. Initial observations received.")

    num_envs=1
    step=0
    games=0
    accum_rewards = np.zeros((num_envs, 2), dtype=np.float32)
    while games < num:
        if not env.agents:
            print("No Agents are available, exiting test.")
            break
            
        # Generate random actions for each agent
        actions = {
            agent: env.action_space(agent).sample() for agent in env.agents
        }
        
        # Step the environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        blue = env.possible_agents[:2]  # ['agent_0','agent_1']
        accum_rewards += np.array([rewards[blue[0]], rewards[blue[1]]], dtype=np.float32)
        step+=1
        
        if np.all(list(truncations.values())):
            print(f"  - Episode {games + 1} finished in {step} steps.")
            step=0
            games+=1
            print("---------------")
            for env_num,rw in enumerate(accum_rewards):
                print(f"rewards for env {env_num}: agent 1={rw[0]} agent 2={rw[1]}")
            accum_rewards.fill(0)
            # Reset is needed to start the next episode
            env.reset()

    #print(f"\nFinished: Ran {games}/{num} episodes.")
    print("PettingZoo environment test finished.\n")
    env.close()

def test_vectorized_env(num_envs=8, total_episodes_to_finish=10):
    """
    Tests the vectorized environment by running until a specific number of episodes
    have been completed across all parallel environments.
    """
    print(f"--- Testing Vectorized Environment (with {num_envs} parallel envs) ---")
    
    # Create the vectorized environment
    vec_env = vectorized_env(num_envs=num_envs)
    
    observations = vec_env.reset()
    print("Reset successful.")
    
    games_completed = 0
    steps = 0

    accum_rewards = np.zeros((num_envs, 2), dtype=np.float32)
    while games_completed < total_episodes_to_finish:
        # For a vectorized environment, we need to sample an action for each parallel environment.
        # The result is a tuple of arrays, where each array has a shape of (num_envs, *action_shape).
        actions = []
        for _ in range(num_envs):
            actions.append(vec_env.action_space.sample())
        
        # This transposes the list of tuples into a tuple of lists, then stacks them.
        # e.g., [(a1, b1), (a2, b2)] -> ([a1, a2], [b1, b2]) -> (array([a1, a2]), array([b1, b2]))
        actions = tuple(np.stack(x) for x in zip(*actions))

        # Step the environment
        observations, rewards, terminations, truncations, infos = vec_env.step(actions)
        accum_rewards += rewards
        
        steps += 1
        
        # Check how many environments finished and add to the count
        # The vectorized env auto-resets on termination/truncation
        #num_finished = np.sum(terminations) + np.sum(truncations)
        #episodes_completed += num_finished

        for env_idx, info in enumerate(infos):
            team = info.get("goal_scored_by")
            if team is not None:
                print(f"Env {env_idx+1}: Goal scored by {team} team!")

        if truncations.all():
            games_completed += 1
            print("---------------")
            for env,rw in enumerate(accum_rewards):
                print(f"rewards for env {env}: agent 1={rw[0]} agent 2={rw[1]}")
            print(f"{games_completed*num_envs} games finished")
            accum_rewards.fill(0)
            observations = vec_env.reset()

        
        # if num_finished > 0:
        #     print(f"Step {steps}: {num_finished} episode(s) finished. Total finished: {episodes_completed}/{total_episodes_to_finish}")

    print(f"\nFinished running. Completed {games_completed*num_envs} episodes in {steps} steps.")
    print("Vectorized environment test finished.")
    vec_env.close()

if __name__ == "__main__":
    # Make sure your simulation server is running before executing this script.
    
    # --- PettingZoo Test ---
    # Run 10 full games (episodes)
    test_pettingzoo_env(num=10)
    
    # --- Vectorized Environment Test ---
    # Run until 20 total episodes have been completed across 8 parallel environments
    test_vectorized_env(num_envs=8, total_episodes_to_finish=20)
