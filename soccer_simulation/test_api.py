import requests
import numpy as np
import time
import json
import argparse

# --- Configuration ---
HOST = "localhost"
PORT = 5000
URL = f"http://{HOST}:{PORT}"
NUM_AGENTS = 4
ACTION_DIM = 3  # (force_x, force_y, torque)

def reset_all_envs(num_envs):
    print("Resetting all environments...")
    try:
        res = requests.post(f"{URL}/reset_all")
        res.raise_for_status()
        # We can ignore the initial observation in this test script
        print("All environments reset successfully.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error resetting all environments: {e}")
        return False

def generate_random_actions(num_envs):
    """Generates a batch of random actions for all agents in all environments as a list."""
    actions_all = []
    for _ in range(num_envs):
        agent_actions = {}
        for agent_idx in range(NUM_AGENTS):
            # Example: [force_x, force_y, torque]
            action = [
                np.random.uniform(-150000, 150000), # Increased force range
                np.random.uniform(-150000, 150000), # Increased force range
                np.random.uniform(-1e5, 1e5)       # Decreased torque range
            ]
            agent_actions[f"agent_{agent_idx}"] = action
        actions_all.append(agent_actions)
    return actions_all

def main():
    parser = argparse.ArgumentParser(description="API Test Client for Soccer Simulation")
    parser.add_argument("--num-envs", type=int, default=8, choices=[1, 8], help="Number of environments to test (1 or 8)")
    args = parser.parse_args()
    
    num_envs = args.num_envs

    if not reset_all_envs(num_envs):
        return
        
    scores = [{"blue": 0, "red": 0} for _ in range(num_envs)]
    accumulated_rewards = np.zeros((num_envs, 2)) # (num_envs, num_blue_agents)

    print(f"\nStarting simulation loop for {num_envs} environment(s)...")
    try:
        step_count = 0
        while True:
            actions = generate_random_actions(num_envs)
            
            try:
                start_time = time.time()
                res = requests.post(f"{URL}/step", json={"actions": actions})
                res.raise_for_status()
                duration = time.time() - start_time
                
                data = res.json()
                
                # Accumulate rewards
                accumulated_rewards += np.array(data["rewards"])

                # Update scores from live score in infos
                for i, info in enumerate(data["infos"]):
                    if "score" in info:
                        scores[i]["blue"] = info["score"].get("blue", scores[i]["blue"])
                        scores[i]["red"] = info["score"].get("red", scores[i]["red"])
                
                # Check if the episode has finished
                if any(data["dones"]):
                    print("\n--- Episode Finished ---")
                    print("Final Scores:")
                    for i, score in enumerate(scores):
                        print(f"  Env {i}: Blue {score['blue']} - Red {score['red']}")
                    
                    print("\nAccumulated Step Rewards (Blue Team):")
                    for i, env_rewards in enumerate(accumulated_rewards):
                        print(f"  Env {i}: Agent 0: {env_rewards[0]:.4f}, Agent 1: {env_rewards[1]:.4f}")

                    print("------------------------\n")
                    
                    if not reset_all_envs(num_envs):
                        break
                        
                    scores = [{"blue": 0, "red": 0} for _ in range(num_envs)]
                    accumulated_rewards.fill(0)
                    step_count = 0
                    continue

                step_count += 1
                
            except requests.exceptions.RequestException as e:
                print(f"Error during step: {e}")
                break
                
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping simulation.")

if __name__ == "__main__":
    main()
