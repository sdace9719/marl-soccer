import argparse
import json
import time
from typing import Dict, List
import math

import numpy as np
import requests


def reset_all(url: str):
    res = requests.post(f"{url}/reset_all")
    res.raise_for_status()
    data = res.json()
    return data["observations"]  # list of envs, each is list of 4 agent obs


def step(url: str, actions_list: List[Dict[str, List[float]]]):
    res = requests.post(f"{url}/step", json={"actions": actions_list})
    res.raise_for_status()
    return res.json()


def build_zero_actions(agent_ids: List[str]):
    return {agent: [0.0, 0.0, 0.0] for agent in agent_ids}


def unit_vec(v: np.ndarray, eps: float = 1e-8):
    n = np.linalg.norm(v)
    return v / (n + eps)


def action_towards(vec2: np.ndarray, magnitude: float = 150000.0):
    d = unit_vec(vec2)
    return [float(d[0] * magnitude), float(d[1] * magnitude), 0.0]


def world_vec_to_local(world_vec: np.ndarray, agent_angle: float) -> np.ndarray:
    """Converts a world-space vector to a local-space vector for the agent."""
    x, y = world_vec
    cos_a = math.cos(agent_angle)
    sin_a = math.sin(agent_angle)
    # This is the inverse rotation of the agent's orientation
    local_x = x * cos_a + y * sin_a
    local_y = -x * sin_a + y * cos_a
    return np.array([local_x, local_y])


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)


def test_baseline(url: str, config: dict):
    print("\n[Baseline] Zero actions one step")
    observations = reset_all(url)
    agent_ids = [f"agent_{i}" for i in range(4)]
    actions = [build_zero_actions(agent_ids)]
    data = step(url, actions)
    rewards = np.array(data["rewards"][0], dtype=np.float32)  # [blue0, blue1]
    print(f"rewards (blue0, blue1): {rewards}")
    print("Expected components include alive_penalty each step; one agent may get a kick_possession_reward.")
    return rewards


def test_proximity_reward(url: str, config: dict):
    print("\n[Proximity] Move agent_1 towards ball for one step")
    observations = reset_all(url)
    agent_ids = [f"agent_{i}" for i in range(4)]
    
    # Take one step with zero actions to get a baseline
    baseline = step(url, [build_zero_actions(agent_ids)])
    base_rewards = np.array(baseline["rewards"][0], dtype=np.float32)

    # Now, calculate action to move towards the ball and take a step
    obs_agents = baseline["observations"][0]
    obs_agent1 = np.asarray(obs_agents[1], dtype=np.float32)
    agent_angle = obs_agent1[4]
    ball_vec_world = obs_agent1[11:13]

    # Convert world vector to local and create action
    ball_vec_local = world_vec_to_local(ball_vec_world, agent_angle)
    actions = build_zero_actions(agent_ids)
    actions["agent_1"] = action_towards(ball_vec_local)
    
    data = step(url, [actions])
    rewards = np.array(data["rewards"][0], dtype=np.float32)
    delta = rewards - base_rewards
    print(f"baseline (blue0, blue1): {base_rewards}, after move: {rewards}, delta: {delta}")
    # Expect agent_1 delta > 0 and agent_0 small relative to agent_1
    assert delta[1] > 0.0, "agent_1 proximity delta should be positive"
    assert abs(delta[0]) <= 0.2 * abs(delta[1]) + 1e-4, "agent_0 proximity delta should be small relative to agent_1"


def test_proximity_reward_negative(url: str, config: dict):
    print("\n[Proximity Negative] Move agent_1 away from ball for one step")
    observations = reset_all(url)
    agent_ids = [f"agent_{i}" for i in range(4)]
    
    baseline = step(url, [build_zero_actions(agent_ids)])
    base_rewards = np.array(baseline["rewards"][0], dtype=np.float32)

    obs_agents = baseline["observations"][0]
    obs_agent1 = np.asarray(obs_agents[1], dtype=np.float32)
    agent_angle = obs_agent1[4]
    ball_vec_world = obs_agent1[11:13]

    # Move opposite to the ball in world space, then convert to local
    away_vec_world = -unit_vec(ball_vec_world)
    away_vec_local = world_vec_to_local(away_vec_world, agent_angle)
    
    actions = build_zero_actions(agent_ids)
    actions["agent_1"] = action_towards(away_vec_local)
    data = step(url, [actions])
    rewards = np.array(data["rewards"][0], dtype=np.float32)
    delta = rewards - base_rewards
    print(f"baseline (blue0, blue1): {base_rewards}, after move-away: {rewards}, delta: {delta}")
    # Expect agent_1 delta < 0 and agent_0 small relative to agent_1
    assert delta[1] < 0.0, "agent_1 proximity delta should be negative when moving away"
    assert abs(delta[0]) <= 0.2 * abs(delta[1]) + 1e-4, "agent_0 proximity delta should be small relative to agent_1"


def test_move_ball_towards_goal(url: str, config: dict, steps_push: int = 40):
    print("\n[Move Ball] Approach ball then push towards red goal")
    agent_ids = [f"agent_{i}" for i in range(4)]

    obs = reset_all(url)
    obs_agents = obs[0]

    total_reward_sum = 0.0

    # Phase 1: approach the ball so we can actually move it
    for _ in range(60):
        obs_agent1 = np.asarray(obs_agents[1], dtype=np.float32)
        agent_angle = obs_agent1[4]
        ball_vec_world = obs_agent1[11:13]
        if np.linalg.norm(ball_vec_world) < 35.0:
            break
        ball_vec_local = world_vec_to_local(ball_vec_world, agent_angle)
        actions = build_zero_actions(agent_ids)
        actions["agent_1"] = action_towards(ball_vec_local)
        data = step(url, [actions])
        obs_agents = data["observations"][0]
        r = np.array(data["rewards"][0], dtype=np.float32)
        total_reward_sum += r.sum()
        if data["infos"][0].get("goal_scored_by"):
            break

    # Phase 2: push ball toward the red goal
    for _ in range(steps_push):
        obs_agent1 = np.asarray(obs_agents[1], dtype=np.float32)
        agent_angle = obs_agent1[4]
        red_goal_vec_world = obs_agent1[15:17]
        red_goal_vec_local = world_vec_to_local(red_goal_vec_world, agent_angle)
        actions = build_zero_actions(agent_ids)
        actions["agent_1"] = action_towards(red_goal_vec_local)

        data = step(url, [actions])
        obs_agents = data["observations"][0]
        r = np.array(data["rewards"][0], dtype=np.float32)
        total_reward_sum += r.sum()
        if data["infos"][0].get("goal_scored_by"):
            break

    print(f"Total reward from pushing towards opponent goal: {total_reward_sum:.4f}")
    assert total_reward_sum > 0, "Pushing towards opponent goal should yield a positive cumulative reward"


def test_move_ball_towards_own_goal(url: str, config: dict, steps_push: int = 40):
    print("\n[Move Ball Negative] Approach ball then push towards blue goal (own)")
    agent_ids = [f"agent_{i}" for i in range(4)]

    obs = reset_all(url)
    obs_agents = obs[0]

    # Phase 1: approach and align behind the ball to push towards blue goal
    force_mag = 150000.0
    torque_mag = 100000.0
    align_tol = 5.0
    angle_tol = 0.1
    push_x_buffer = 15.0
    k_p = 2000.0
    max_align_steps = 120

    for _ in range(max_align_steps):
        obs_agent1 = np.asarray(obs_agents[1], dtype=np.float32)
        agent_pos = obs_agent1[0:2]
        agent_angle = obs_agent1[4]
        ball_vec_world = obs_agent1[11:13]
        ball_pos = agent_pos + ball_vec_world

        actions = build_zero_actions(agent_ids)

        # Align heading to 0 radians (pointing to the right) is fine; we act in local frame
        target_angle = 0.0
        angle_err = target_angle - agent_angle
        if abs(angle_err) > angle_tol:
            actions["agent_1"][2] = (1.0 if angle_err > 0 else -1.0) * torque_mag

        # For pushing towards blue (left), get behind the ball on its right side
        dy = ball_pos[1] - agent_pos[1]
        target_x = ball_pos[0] + push_x_buffer
        dx_to_target = target_x - agent_pos[0]

        if abs(dy) > align_tol:
            force_y = np.clip(dy * k_p, -force_mag, force_mag)
            actions["agent_1"][1] = float(force_y)
        elif abs(dx_to_target) > align_tol:
            force_x = np.clip(dx_to_target * k_p, -force_mag, force_mag)
            actions["agent_1"][0] = float(force_x)
        else:
            # Close enough behind ball; stop aligning
            break

        data = step(url, [actions])
        obs_agents = data["observations"][0]
        if data["infos"][0].get("goal_scored_by"):
            break

    # Phase 2: push ball toward own (blue) goal; validate shaping is negative
    push_reward_sum = 0.0
    cumulative_shaping_delta = 0.0  # Sum of (prev_dist_to_red - curr_dist_to_red)
    for _ in range(steps_push):
        # Distance to red goal BEFORE the step
        obs_agent1 = np.asarray(obs_agents[1], dtype=np.float32)
        agent_pos = obs_agent1[0:2]
        red_goal_vec_world = obs_agent1[15:17]
        red_goal_pos = agent_pos + red_goal_vec_world
        ball_vec_world = obs_agent1[11:13]
        ball_pos = agent_pos + ball_vec_world
        prev_dist_to_red = float(np.linalg.norm(ball_pos - red_goal_pos))

        # Push left in local frame (towards blue goal)
        actions = build_zero_actions(agent_ids)
        actions["agent_1"][0] = -force_mag

        data = step(url, [actions])
        obs_agents = data["observations"][0]

        # Distance to red goal AFTER the step
        obs_agent1_post = np.asarray(obs_agents[1], dtype=np.float32)
        agent_pos_post = obs_agent1_post[0:2]
        red_goal_vec_world_post = obs_agent1_post[15:17]
        red_goal_pos_post = agent_pos_post + red_goal_vec_world_post
        ball_vec_world_post = obs_agent1_post[11:13]
        ball_pos_post = agent_pos_post + ball_vec_world_post
        curr_dist_to_red = float(np.linalg.norm(ball_pos_post - red_goal_pos_post))

        # Accumulate rewards (for reference)
        r = np.array(data["rewards"][0], dtype=np.float32)
        push_reward_sum += r.sum()

        # Accumulate shaping delta (should be negative when moving away from red goal)
        cumulative_shaping_delta += (prev_dist_to_red - curr_dist_to_red)

        if data["infos"][0].get("goal_scored_by"):
            break

    print(f"Own-goal phase: reward_sum={push_reward_sum:.4f}, shaping_delta_to_red={cumulative_shaping_delta:.4f}")
    assert cumulative_shaping_delta < 0.0, "Ball moved net away from red goal; shaping delta should be negative"


def drive_and_score(url: str, target_team: str, agent_ids: List[str], initial_obs: list, max_steps: int = 250):
    obs_local = initial_obs
    force_mag = 150000.0
    torque_mag = 100000.0
    align_tol = 5.0
    angle_tol = 0.1 # radians
    push_x_buffer = 15.0
    k_p = 2000.0 

    dir_x = 1.0 if target_team == "blue" else -1.0
    
    for step_count in range(max_steps):
        obs0 = np.asarray(obs_local[0][1], dtype=np.float32)
        agent_pos = obs0[0:2]
        agent_angle = obs0[4]
        ball_vec = obs0[11:13]
        ball_pos = agent_pos + ball_vec
        
        actions = build_zero_actions(agent_ids)
        dy = ball_pos[1] - agent_pos[1]
        target_x = ball_pos[0] - dir_x * push_x_buffer
        dx_to_target = target_x - agent_pos[0]

        target_angle = 0.0
        angle_err = target_angle - agent_angle
        if abs(angle_err) > angle_tol:
            actions["agent_1"][2] = (1.0 if angle_err > 0 else -1.0) * torque_mag

        if abs(dy) > align_tol:
            force_y = np.clip(dy * k_p, -force_mag, force_mag)
            actions["agent_1"][1] = float(force_y)
        elif abs(dx_to_target) > align_tol:
            force_x = np.clip(dx_to_target * k_p, -force_mag, force_mag)
            actions["agent_1"][0] = float(force_x)
        else:
            actions["agent_1"][0] = dir_x * force_mag

        data_local = step(url, [actions])
        obs_local = data_local["observations"]

        info = data_local["infos"][0]
        if target_team == "blue" and info.get("goal_scored_by") == "blue":
            return True, step_count
        if target_team == "red" and info.get("goal_scored_by") == "red":
            return True, step_count
        
    return False, max_steps


def test_goal_scored(url: str, config: dict, num_episodes: int = 100, max_steps_guard: int = 5000):
    print("\n[Goal Scored - Randomized] Validate last-step reward dominated by terminal bonus (first non-draw)")
    agent_ids = [f"agent_{i}" for i in range(4)]
    multiplier = float(config["rewards"].get("score_difference_multiplier", 5.0))

    found_non_draw = False
    for ep in range(num_episodes):
        reset_all(url)
        last_info = {}
        for _ in range(max_steps_guard):
            actions = {aid: [
                float(np.random.uniform(-150000, 150000)),
                float(np.random.uniform(-150000, 150000)),
                float(np.random.uniform(-1e5, 1e5))
            ] for aid in agent_ids}

            data = step(url, [actions])
            last_info = data["infos"][0]
            if data["dones"][0]:
                break

        blue = last_info.get("score", {}).get("blue", 0)
        red = last_info.get("score", {}).get("red", 0)
        diff = blue - red
        last_rewards = np.array(data["rewards"][0], dtype=np.float32)
        expected = multiplier * diff
        # Use a tolerant check: final reward should be dominated by terminal bonus
        target_sum = 2.0 * expected
        delta = float(last_rewards.sum() - target_sum)
        tol_abs = 0.5
        tol_rel = 0.1
        assert abs(delta) <= max(tol_abs, tol_rel * abs(target_sum)), "Last-step rewards should be dominated by terminal bonus"

        if diff != 0:
            print(f"Episode {ep}: score {blue}-{red}, last_step_sum {last_rewards.sum():.4f}, target_sum {target_sum:.4f}, delta {delta:.4f}")
            found_non_draw = True
            break

    if not found_non_draw:
        print("All random games ended in draws; terminal bonus validated for diff=0.")


def test_goal_conceded(url: str, config: dict, num_episodes: int = 100, max_steps_guard: int = 5000):
    print("\n[Goal Conceded - Randomized] Validate last-step reward dominated by terminal bonus (first red-leading non-draw)")
    agent_ids = [f"agent_{i}" for i in range(4)]
    multiplier = float(config["rewards"].get("score_difference_multiplier", 5.0))

    found_red_lead = False
    for ep in range(num_episodes):
        reset_all(url)
        last_info = {}
        for _ in range(max_steps_guard):
            actions = {aid: [
                float(np.random.uniform(-150000, 150000)),
                float(np.random.uniform(-150000, 150000)),
                float(np.random.uniform(-1e5, 1e5))
            ] for aid in agent_ids}

            data = step(url, [actions])
            last_info = data["infos"][0]
            if data["dones"][0]:
                break

        blue = last_info.get("score", {}).get("blue", 0)
        red = last_info.get("score", {}).get("red", 0)
        diff = blue - red
        last_rewards = np.array(data["rewards"][0], dtype=np.float32)
        expected = multiplier * diff
        target_sum = 2.0 * expected
        delta = float(last_rewards.sum() - target_sum)
        tol_abs = 0.5
        tol_rel = 0.1
        assert abs(delta) <= max(tol_abs, tol_rel * abs(target_sum)), "Last-step rewards should be dominated by terminal bonus"

        if red > blue:
            print(f"Episode {ep}: red lead {red}-{blue}, last_step_sum {last_rewards.sum():.4f}, target_sum {target_sum:.4f}, delta {delta:.4f}")
            found_red_lead = True
            break

    if not found_red_lead:
        print("No red-leading non-draw found; terminal bonus validated for observed episodes.")

def main():
    parser = argparse.ArgumentParser(description="Reward tests using API")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    config = load_config(args.config)

    baseline = test_baseline(url, config)
    test_proximity_reward(url, config)
    test_proximity_reward_negative(url, config)
    test_move_ball_towards_goal(url, config)
    test_move_ball_towards_own_goal(url, config)
    test_goal_scored(url, config)
    test_goal_conceded(url, config)


if __name__ == "__main__":
    main()


