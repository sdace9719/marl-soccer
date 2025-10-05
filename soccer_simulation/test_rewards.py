import argparse
import os
import sys
# Ensure project root is on sys.path so 'soccer_simulation.*' imports resolve when running this file directly
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import time
from typing import Dict, List, Optional
import math
from game.constants import SCREEN_WIDTH, SCREEN_HEIGHT, FIELD_MARGIN

import numpy as np

from soccer_env import soccerenv, get_observation_scalers


_ENV = None  # type: Optional[object]
_SCALERS = None  # type: Optional[dict]
_RENDER = False
_EVAL_DELAY_S = 0.0


def _get_env():
    global _ENV
    global _SCALERS
    if _ENV is None:
        _ENV = soccerenv(render_mode=("human" if _RENDER else None))
        _SCALERS = get_observation_scalers(_ENV)
    return _ENV


def _obs_dict_to_list(obs_dict: Dict[str, np.ndarray]):
    keys = [f"agent_{i}" for i in range(4)]
    return [np.asarray(obs_dict[k], dtype=np.float32) for k in keys]

# Helper to get latest frame slice from stacked obs (3 * frame_size)
FRAME_SIZE = 22
STACK_SIZE = 3

def latest_frame(obs_vec: np.ndarray) -> np.ndarray:
    return obs_vec[(STACK_SIZE - 1) * FRAME_SIZE : STACK_SIZE * FRAME_SIZE]

# Per-frame indices for unit+mag encoding
ANG_IDX = 2
ANG_VEL_IDX = 3
TEAMMATE_START = 4
OPP1_START = 7
OPP2_START = 10
BALL_START = 13
OWN_GOAL_START = 16
OPP_GOAL_START = 19

def vec_from(obs_frame: np.ndarray, start_idx: int) -> np.ndarray:
    unit = obs_frame[start_idx:start_idx+2]
    mag = obs_frame[start_idx+2]
    # rescale magnitude back to world units using field diagonal
    field_diag = float(_SCALERS["field_diagonal"]) if _SCALERS is not None else 1.0
    return unit * (mag * field_diag)


def reset_all(url: str):
    env = _get_env()
    obs_dict, infos = env.reset()
    obs_list = _obs_dict_to_list(obs_dict)
    if _RENDER:
        env.render()
        if _EVAL_DELAY_S and _EVAL_DELAY_S > 0:
            time.sleep(min(float(_EVAL_DELAY_S), 5.0))
    return [obs_list]  # list of envs, each is list of 4 agent obs


def step(url: str, actions_list: List[Dict[str, List[float]]]):
    env = _get_env()
    # Single-env adapter: take the first actions dict
    actions = actions_list[0]
    obs_dict, rewards_dict, terminations, truncations, infos = env.step(actions)
    if _RENDER:
        env.render()
        if _EVAL_DELAY_S and _EVAL_DELAY_S > 0:
            time.sleep(min(float(_EVAL_DELAY_S), 5.0))
    obs_list = _obs_dict_to_list(obs_dict)
    # Rewards are only for blue agents (agent_0, agent_1)
    rewards_list = [float(rewards_dict.get("agent_0", 0.0)), float(rewards_dict.get("agent_1", 0.0))]
    done = bool(any(terminations.values()) or any(truncations.values()))
    # All agents carry identical info; pick one
    info_env = {}
    if len(infos) > 0:
        info_env = infos.get("agent_0", {})
    return {
        "observations": [obs_list],
        "rewards": [rewards_list],
        "dones": [done],
        "infos": [info_env],
    }


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


def test_proximity_reward_for_agent(url: str, config: dict, agent_idx: int):
    print(f"\n[Proximity] Move agent_{agent_idx} towards ball for one step")
    observations = reset_all(url)
    agent_ids = [f"agent_{i}" for i in range(4)]
    
    # Take one step with zero actions to get a baseline
    baseline = step(url, [build_zero_actions(agent_ids)])
    base_rewards = np.array(baseline["rewards"][0], dtype=np.float32)

    # Perform multiple consecutive steps towards the ball to amplify the effect
    N_STEPS = 6
    sum_rewards = np.zeros_like(base_rewards)
    obs_agents = baseline["observations"][0]
    for _ in range(N_STEPS):
        obs_agent = np.asarray(obs_agents[agent_idx], dtype=np.float32)
        obs_agent = latest_frame(obs_agent)
        agent_angle = float(obs_agent[ANG_IDX] * math.pi)
        ball_vec_world = vec_from(obs_agent, BALL_START)
        ball_vec_local = world_vec_to_local(ball_vec_world, agent_angle)
        actions = build_zero_actions(agent_ids)
        actions[f"agent_{agent_idx}"] = action_towards(ball_vec_local)
        data = step(url, [actions])
        rewards = np.array(data["rewards"][0], dtype=np.float32)
        sum_rewards += rewards
        obs_agents = data["observations"][0]

    delta = sum_rewards - base_rewards * N_STEPS
    print(f"baseline (blue0, blue1): {base_rewards}, after N={N_STEPS} steps sum: {sum_rewards}, delta: {delta}")
    # Expect moved agent delta > 0; do not compare other agent
    assert delta[agent_idx] > 0.0, f"agent_{agent_idx} proximity delta should be positive"


def test_proximity_reward_negative(url: str, config: dict):
    agent_ids = [f"agent_{i}" for i in range(4)]
    for agent_idx in (0, 1):
        print(f"\n[Proximity Negative] Move agent_{agent_idx} away from ball for one step")
        observations = reset_all(url)
        baseline = step(url, [build_zero_actions(agent_ids)])
        base_rewards = np.array(baseline["rewards"][0], dtype=np.float32)

        # Perform multiple consecutive steps away from the ball to amplify the effect
        N_STEPS = 6
        sum_rewards = np.zeros_like(base_rewards)
        obs_agents = baseline["observations"][0]
        for _ in range(N_STEPS):
            obs_agent = np.asarray(obs_agents[agent_idx], dtype=np.float32)
            obs_agent = latest_frame(obs_agent)
            agent_angle = float(obs_agent[ANG_IDX] * math.pi)
            ball_vec_world = vec_from(obs_agent, BALL_START)
            away_vec_world = -unit_vec(ball_vec_world)
            away_vec_local = world_vec_to_local(away_vec_world, agent_angle)
            actions = build_zero_actions(agent_ids)
            actions[f"agent_{agent_idx}"] = action_towards(away_vec_local)
            data = step(url, [actions])
            rewards = np.array(data["rewards"][0], dtype=np.float32)
            sum_rewards += rewards
            obs_agents = data["observations"][0]

        delta = sum_rewards - base_rewards * N_STEPS
        print(f"baseline (blue0, blue1): {base_rewards}, after N={N_STEPS} steps sum: {sum_rewards}, delta: {delta}")
        assert delta[agent_idx] < 0.0, f"agent_{agent_idx} proximity delta should be negative when moving away"


def test_move_ball_towards_goal_for_agent(url: str, config: dict, agent_idx: int, steps_push: int = 5):
    print(f"\n[Move Ball] agent_{agent_idx}: Approach ball then push towards red goal ({steps_push} steps)")
    agent_ids = [f"agent_{i}" for i in range(4)]

    obs = reset_all(url)
    obs_agents = obs[0]

    total_reward_sum = 0.0

    # Phase 1: approach the ball so we can actually move it (single episode guard)
    for _ in range(60):
        obs_agent = np.asarray(obs_agents[agent_idx], dtype=np.float32)
        obs_agent = latest_frame(obs_agent)
        agent_angle = float(obs_agent[ANG_IDX] * math.pi)
        ball_vec_world = vec_from(obs_agent, BALL_START)
        if np.linalg.norm(ball_vec_world) < 35.0:
            break
        ball_vec_local = world_vec_to_local(ball_vec_world, agent_angle)
        actions = build_zero_actions(agent_ids)
        actions[f"agent_{agent_idx}"] = action_towards(ball_vec_local)
        data = step(url, [actions])
        obs_agents = data["observations"][0]
        if data["dones"][0]:
            raise AssertionError("Episode ended during approach phase in move-ball-towards-goal test")
        r = np.array(data["rewards"][0], dtype=np.float32)
        total_reward_sum += r.sum()
        if data["infos"][0].get("goal_scored_by"):
            raise AssertionError("A goal was scored during approach phase; keep within a single continuous episode")

    # Phase 2: push ball toward the red goal
    for _ in range(steps_push):
        obs_agent = np.asarray(obs_agents[agent_idx], dtype=np.float32)
        obs_agent = latest_frame(obs_agent)
        agent_angle = float(obs_agent[ANG_IDX] * math.pi)
        red_goal_vec_world = vec_from(obs_agent, OPP_GOAL_START)
        red_goal_vec_local = world_vec_to_local(red_goal_vec_world, agent_angle)
        actions = build_zero_actions(agent_ids)
        actions[f"agent_{agent_idx}"] = action_towards(red_goal_vec_local)

        data = step(url, [actions])
        obs_agents = data["observations"][0]
        if data["dones"][0]:
            raise AssertionError("Episode ended during push phase in move-ball-towards-goal test")
        r = np.array(data["rewards"][0], dtype=np.float32)
        total_reward_sum += r.sum()
        if data["infos"][0].get("goal_scored_by"):
            raise AssertionError("A goal was scored during push phase; keep within a single continuous episode")

    print(f"Total reward from pushing towards opponent goal: {total_reward_sum:.4f}")
    assert total_reward_sum > 0, "Pushing towards opponent goal should yield a positive cumulative reward"


def test_move_ball_towards_own_goal_for_agent(url: str, config: dict, agent_idx: int, steps_push: int = 40):
    print(f"\n[Move Ball Negative] agent_{agent_idx}: Approach ball then push towards blue goal (own)")
    agent_ids = [f"agent_{i}" for i in range(4)]

    # Temporarily enable rendering for this test
    global _ENV, _RENDER, _EVAL_DELAY_S
    prev_render = _RENDER
    prev_delay = _EVAL_DELAY_S
    _RENDER = True
    _EVAL_DELAY_S = 0.01  # 10ms
    # Recreate env with human render
    if _ENV is not None:
        try:
            _ENV.close()
        except Exception:
            pass
        _ENV = None

    obs = reset_all(url)
    obs_agents = obs[0]

    # Phase 1: move only in X to get ahead of the ball, then align Y
    force_mag = 150000.0
    k_p = 2000.0
    align_tol = 5.0
    ahead_buffer_x = 15.0

    # X-only alignment: ensure agent is ahead of the ball in +x (so pushing left scores own goal)
    while True:
        obs_agent = latest_frame(np.asarray(obs_agents[agent_idx], dtype=np.float32))
        ball_vec_world = vec_from(obs_agent, BALL_START)
        # Agent ahead means ball is to the left of agent by a buffer: ball_vec_world.x < -buffer
        if float(ball_vec_world[0]) < -ahead_buffer_x:
            break
        actions = build_zero_actions(agent_ids)
        # Only apply X force; keep Y at 0 for this phase
        actions[f"agent_{agent_idx}"][0] = force_mag
        data = step(url, [actions])
        obs_agents = data["observations"][0]
        if data["dones"][0]:
            raise AssertionError("Episode ended during X alignment in move-ball-towards-own-goal test")

    # Y-only alignment: simple low-magnitude force toward ball's y
    y_align_tol = 1.0
    y_force_norm = 0.15  # small normalized force for smooth convergence
    while True:
        obs_agent = latest_frame(np.asarray(obs_agents[agent_idx], dtype=np.float32))
        ball_vec_world = vec_from(obs_agent, BALL_START)
        dy = float(ball_vec_world[1])
        if abs(dy) <= y_align_tol:
            break
        actions = build_zero_actions(agent_ids)
        # Only apply Y force; keep X at 0 for this phase
        actions[f"agent_{agent_idx}"][1] = float(np.sign(dy) * y_force_norm)
        data = step(url, [actions])
        obs_agents = data["observations"][0]
        if data["dones"][0]:
            raise AssertionError("Episode ended during Y alignment in move-ball-towards-own-goal test")

    # Phase 2: push toward own goal until a red goal is scored (no step limit)
    push_reward_sum = 0.0
    cumulative_shaping_delta = 0.0  # Sum of (prev_dist_to_red - curr_dist_to_red)
    goal_scored = False
    while not goal_scored:
        obs_agent = latest_frame(np.asarray(obs_agents[agent_idx], dtype=np.float32))
        agent_angle = float(obs_agent[ANG_IDX] * math.pi)
        ball_vec_world = vec_from(obs_agent, BALL_START)
        red_goal_vec_world = vec_from(obs_agent, OPP_GOAL_START)  # opponent goal (red)
        # Distance from ball to red goal = || ball_vec - red_goal_vec ||
        prev_dist_to_red = float(np.linalg.norm(ball_vec_world - red_goal_vec_world))

        # Push hard left toward own goal with slight vertical tracking
        # This applies maximum leftward force and a small Y component to keep contact
        actions = build_zero_actions(agent_ids)
        unit = unit_vec(ball_vec_world)
        actions[f"agent_{agent_idx}"][0] = -1.0
        actions[f"agent_{agent_idx}"][1] = float(np.clip(unit[1] * 0.5, -0.5, 0.5))

        data = step(url, [actions])
        obs_agents = data["observations"][0]

        obs_post = latest_frame(np.asarray(obs_agents[agent_idx], dtype=np.float32))
        ball_vec_world_post = vec_from(obs_post, BALL_START)
        red_goal_vec_world_post = vec_from(obs_post, OPP_GOAL_START)
        curr_dist_to_red = float(np.linalg.norm(ball_vec_world_post - red_goal_vec_world_post))

        r = np.array(data["rewards"][0], dtype=np.float32)
        push_reward_sum += r.sum()
        cumulative_shaping_delta += (prev_dist_to_red - curr_dist_to_red)
        if data["infos"][0].get("goal_scored_by") == "red":
            goal_scored = True
            break
        if data["dones"][0]:
            # Episode ended before conceding; stop the test gracefully
            break

    print(f"Own-goal phase (episode end): reward_sum={push_reward_sum:.4f}, shaping_delta_to_red={cumulative_shaping_delta:.4f}")
    assert push_reward_sum < 0.0, "Episode cumulative reward should be negative when pushing towards own goal"

    # Restore previous render settings
    try:
        if _ENV is not None:
            try:
                _ENV.close()
            except Exception:
                pass
            _ENV = None
    finally:
        _RENDER = prev_render
        _EVAL_DELAY_S = prev_delay


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
        obs0 = latest_frame(obs0)
        agent_angle = obs0[2]
        ball_vec = vec_from(obs0, BALL_START)
        ball_pos = obs0[0:2] + ball_vec
        
        actions = build_zero_actions(agent_ids)
        dy = ball_pos[1] - obs0[1]
        target_x = ball_pos[0] - dir_x * push_x_buffer
        dx_to_target = target_x - obs0[0]

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


def test_goal_scored(url: str, config: dict):
    agent_ids = [f"agent_{i}" for i in range(4)]
    multiplier = float(config["rewards"].get("score_difference_multiplier", 5.0))

    for agent_idx in (0, 1):
        reset_all(url)
        obs = step(url, [build_zero_actions(agent_ids)])
        obs_agents = obs["observations"][0]
        total_reward_sum = 0.0

        # Phase 1: align Y with ball (proportional control with capped force)
        y_tol = 3.0
        max_align_steps = 400
        k_p = 2000.0
        force_cap = 50000.0
        for _ in range(max_align_steps):
            obs_agent = np.asarray(obs_agents[agent_idx], dtype=np.float32)
            obs_agent = latest_frame(obs_agent)
            agent_angle = obs_agent[2]
            ball_vec_world = vec_from(obs_agent, BALL_START)
            ball_pos = obs_agent[0:2] + ball_vec_world
            dy = ball_pos[1] - obs_agent[1]
            if abs(dy) <= y_tol:
                break
            # Proportional correction on Y with cap to avoid oscillation
            desired_world = np.array([0.0, dy], dtype=np.float32)
            desired_local = world_vec_to_local(desired_world, agent_angle)
            norm = float(np.linalg.norm(desired_local))
            dir_local = desired_local / norm if norm > 1e-6 else np.array([0.0, 0.0], dtype=np.float32)
            mag = float(np.clip(k_p * abs(dy), -force_cap, force_cap))
            force_local = dir_local * mag
            actions = build_zero_actions(agent_ids)
            actions[f"agent_{agent_idx}"][0] = float(force_local[0])
            actions[f"agent_{agent_idx}"][1] = float(force_local[1])
            data = step(url, [actions])
            obs_agents = data["observations"][0]
            r = np.array(data["rewards"][0], dtype=np.float32)
            total_reward_sum += r.sum()
            if data["dones"][0]:
                raise AssertionError("Episode ended during Y-align for goal scoring")

        # Proceed even if a small residual y offset remains
        obs_agent = np.asarray(obs_agents[agent_idx], dtype=np.float32)
        obs_agent = latest_frame(obs_agent)
        agent_angle = obs_agent[2]
        ball_vec_world = vec_from(obs_agent, BALL_START)
        ball_pos = obs_agent[0:2] + ball_vec_world

        # Phase 2: push toward ball until a goal is scored (expect blue to score) with strong force
        goal_scored = False
        last_info = {}
        for _ in range(3000):
            obs_agent = np.asarray(obs_agents[agent_idx], dtype=np.float32)
            obs_agent = latest_frame(obs_agent)
            ball_vec_world = vec_from(obs_agent, BALL_START)
            ball_pos = obs_agent[0:2] + ball_vec_world
            to_ball_world = ball_pos - obs_agent[0:2]
            # Use world-frame unit direction with boosted magnitude (clipped to [-1,1])
            u = unit_vec(to_ball_world)
            actions = build_zero_actions(agent_ids)
            actions[f"agent_{agent_idx}"][0] = float(np.clip(u[0] * 1.5, -1.0, 1.0))
            actions[f"agent_{agent_idx}"][1] = float(np.clip(u[1] * 1.5, -1.0, 1.0))
            data = step(url, [actions])
            obs_agents = data["observations"][0]
            last_info = data["infos"][0]
            r = np.array(data["rewards"][0], dtype=np.float32)
            total_reward_sum += r.sum()
            if last_info.get("goal_scored_by") == "blue":
                goal_scored = True
                break
            if data["dones"][0]:
                # One extra no-op step to catch goal info if it landed on terminal frame
                data2 = step(url, [build_zero_actions(agent_ids)])
                if data2["infos"][0].get("goal_scored_by") == "blue":
                    goal_scored = True
                    break
                raise AssertionError("Episode ended before scoring a goal")

        assert goal_scored, "Did not score a blue goal deterministically"

        # After goal, do nothing until episode end; check terminal rewards
        last_rewards = None
        while True:
            data = step(url, [build_zero_actions(agent_ids)])
            last_rewards = np.array(data["rewards"][0], dtype=np.float32)
            total_reward_sum += last_rewards.sum()
            if data["dones"][0]:
                break

        final_info = data["infos"][0]
        blue = final_info.get("score", {}).get("blue", 0)
        red = final_info.get("score", {}).get("red", 0)
        diff = blue - red
        target_sum = 2.0 * multiplier * diff
        delta = float(last_rewards.sum() - target_sum)
        tol_abs = 0.5
        tol_rel = 0.1
        print(f"[Goal Scored] agent_{agent_idx}: episode_total_reward={total_reward_sum:.4f}, final_step_sum={last_rewards.sum():.4f}, target_sum={target_sum:.4f}, delta={delta:.4f}")
        assert abs(delta) <= max(tol_abs, tol_rel * abs(target_sum)), "Terminal rewards should reflect score difference (goal scored)"


def test_goal_conceded(url: str, config: dict):
    print("\n[Goal Conceded - Deterministic] Set position past midline, align Y, push to concede (agents 0 and 1)")
    agent_ids = [f"agent_{i}" for i in range(4)]
    multiplier = float(config["rewards"].get("score_difference_multiplier", 5.0))

    for agent_idx in (0, 1):
        reset_all(url)
        obs = step(url, [build_zero_actions(agent_ids)])
        obs_agents = obs["observations"][0]
        total_reward_sum = 0.0

        # Phase 1: move until ball unit x flips (agent ahead), then a few extra steps for clearance
        max_steps_phase1 = 400
        x_flip_tol = -0.05
        extra_after_flip = 15
        flipped = False
        extra = extra_after_flip
        for _ in range(max_steps_phase1):
            obs_agent = latest_frame(np.asarray(obs_agents[agent_idx], dtype=np.float32))
            agent_angle = float(obs_agent[ANG_IDX] * math.pi)
            ball_unit = obs_agent[BALL_START:BALL_START+2]
            if not flipped and float(ball_unit[0]) <= x_flip_tol:
                flipped = True
            if flipped:
                if extra <= 0:
                    break
                extra -= 1
            move_world = np.array([+1.0, 0.0], dtype=np.float32)
            move_local = world_vec_to_local(move_world, agent_angle)
            actions = build_zero_actions(agent_ids)
            actions[f"agent_{agent_idx}"] = action_towards(move_local)
            data = step(url, [actions])
            obs_agents = data["observations"][0]
            if data["dones"][0]:
                raise AssertionError("Episode ended while positioning for concede")

        # Phase 2: align Y so ball unit y ~ 0 using only unit vector
        y_unit_tol = 0.05
        max_align_steps = 400
        for _ in range(max_align_steps):
            obs_agent = latest_frame(np.asarray(obs_agents[agent_idx], dtype=np.float32))
            agent_angle = float(obs_agent[ANG_IDX] * math.pi)
            ball_unit = obs_agent[BALL_START:BALL_START+2]
            if abs(float(ball_unit[1])) <= y_unit_tol:
                break
            move_world = np.array([0.0, np.sign(ball_unit[1])], dtype=np.float32)
            move_local = world_vec_to_local(move_world, agent_angle)
            actions = build_zero_actions(agent_ids)
            actions[f"agent_{agent_idx}"] = action_towards(move_local)
            data = step(url, [actions])
            obs_agents = data["observations"][0]
            if data["dones"][0]:
                raise AssertionError("Episode ended during Y-align for concede")

        # Phase 3: push toward ball until a goal is scored (expect red to score)
        goal_scored = False
        last_info = {}
        for _ in range(2000):
            obs_agent = latest_frame(np.asarray(obs_agents[agent_idx], dtype=np.float32))
            agent_angle = float(obs_agent[ANG_IDX] * math.pi)
            ball_vec_world = vec_from(obs_agent, BALL_START)
            to_ball_world = ball_vec_world
            to_ball_local = world_vec_to_local(to_ball_world, agent_angle)
            actions = build_zero_actions(agent_ids)
            actions[f"agent_{agent_idx}"] = action_towards(to_ball_local)
            data = step(url, [actions])
            obs_agents = data["observations"][0]
            last_info = data["infos"][0]
            r = np.array(data["rewards"][0], dtype=np.float32)
            total_reward_sum += r.sum()
            if last_info.get("goal_scored_by") == "red":
                goal_scored = True
                break
            if data["dones"][0]:
                raise AssertionError("Episode ended before conceding a goal")

        assert goal_scored, "Did not concede a red goal deterministically"

        # After goal, do nothing until episode end; check terminal rewards
        last_rewards = None
        while True:
            data = step(url, [build_zero_actions(agent_ids)])
            last_rewards = np.array(data["rewards"][0], dtype=np.float32)
            total_reward_sum += last_rewards.sum()
            if data["dones"][0]:
                break

        final_info = data["infos"][0]
        blue = final_info.get("score", {}).get("blue", 0)
        red = final_info.get("score", {}).get("red", 0)
        diff = blue - red
        target_sum = 2.0 * multiplier * diff
        delta = float(last_rewards.sum() - target_sum)
        tol_abs = 0.5
        tol_rel = 0.1
        print(f"[Goal Conceded] agent_{agent_idx}: episode_total_reward={total_reward_sum:.4f}, final_step_sum={last_rewards.sum():.4f}, target_sum={target_sum:.4f}, delta={delta:.4f}")
        assert abs(delta) <= max(tol_abs, tol_rel * abs(target_sum)), "Terminal rewards should reflect score difference (goal conceded)"

def main():
    global _RENDER, _EVAL_DELAY_S
    parser = argparse.ArgumentParser(description="Reward tests using self-contained PettingZoo env")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--render", action="store_true", help="Enable graphical rendering")
    parser.add_argument("--eval-delay", type=float, default=0.0, help="Optional delay per API step when rendering (seconds)")
    args = parser.parse_args()

    _RENDER = bool(args.render)
    _EVAL_DELAY_S = float(args.eval_delay)

    # url retained for backward-compatible signatures, but unused in local mode
    url = "local://env"
    config = load_config(args.config)

    baseline = test_baseline(url, config)
    test_proximity_reward_for_agent(url, config, 0)
    test_proximity_reward_negative(url, config)
    test_move_ball_towards_goal_for_agent(url, config, 0)
    test_move_ball_towards_own_goal_for_agent(url, config, 0)
    test_goal_scored(url, config)
    test_goal_conceded(url, config)


if __name__ == "__main__":
    main()


