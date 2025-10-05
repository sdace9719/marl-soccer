from flask import Flask, jsonify, request
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import logging
import time

from soccer_simulation.game.game import Game

# --- Constants ---
NUM_AGENTS = 4
ACTION_DIM = 3
FRAME_SKIPS = 6 # The number of physics ticks per agent action

# --- Globals ---
environments = {}
env_locks = {}
executor = ThreadPoolExecutor()
DEFAULT_EVAL_DELAY_S = 0.0

def create_app():
    app = Flask(__name__)

    def _serialize_batch_obs(batch_obs):
        return [[obs.tolist() for obs in env_obs] for env_obs in batch_obs]

    @app.route('/reset_all', methods=['POST'])
    def reset_all():
        num_envs = len(environments)
        batch_obs = [[] for _ in range(num_envs)]
        for env_id in range(num_envs):
            with env_locks[env_id]:
                obs = environments[env_id].reset()
                batch_obs[env_id] = obs
        return jsonify({"observations": _serialize_batch_obs(batch_obs)})

    def _step_env(env_id, actions):
        """
        Runs the simulation for a fixed number of ticks (frame skipping)
        for a single environment. This loop runs as fast as possible.
        """
        with env_locks[env_id]:
            game = environments[env_id]
            
            # Rewards are accumulated over the skipped frames.
            total_rewards = np.zeros(2) # For the 2 blue agents
            obs, done = None, False
            aggregated_info = {} # The aggregated info dict for the entire burst

            for t in range(FRAME_SKIPS):
                # Apply the chosen action once, then let physics evolve naturally
                # for the remaining skipped frames (forces like friction/damping act).
                step_actions = actions if t == 0 else None
                obs, rewards, done, info = game.step(step_actions)
                total_rewards += np.array(rewards)

                # Always carry forward latest info (e.g., live score). If multiple ticks
                # occur within the burst, the last update wins, reflecting most recent state.
                aggregated_info.update(info)

                # If the episode ends mid-burst, stop early.
                if done:
                    break
            
            return obs, total_rewards.tolist(), done, aggregated_info

    @app.route('/step', methods=['POST'])
    def step():
        actions_list = request.json.get('actions')
        # Optional evaluation delay (in seconds). Allows slow playback for visualization.
        eval_delay_s = 0.0
        try:
            eval_delay_s = float(request.json.get('eval_delay_s', 0.0))
        except Exception:
            eval_delay_s = 0.0
        if eval_delay_s <= 0.0:
            eval_delay_s = DEFAULT_EVAL_DELAY_S
        # --- Basic Input Validation ---
        num_envs = len(environments)
        if not isinstance(actions_list, list) or len(actions_list) != num_envs:
            return jsonify({"error": "Invalid 'actions' format."}), 400
        
        # --- Execute Steps in Parallel ---
        futures = {
            env_id: executor.submit(_step_env, env_id, actions)
            for env_id, actions in enumerate(actions_list)
        }
        
        batch_obs = [[] for _ in range(num_envs)]
        batch_rewards = [[] for _ in range(num_envs)]
        batch_dones = [False for _ in range(num_envs)]
        batch_infos = [{} for _ in range(num_envs)]

        for env_id, future in futures.items():
            obs, rewards, done, info = future.result()
            batch_obs[env_id] = obs
            batch_rewards[env_id] = rewards
            batch_dones[env_id] = done
            batch_infos[env_id] = info

        # Apply eval delay once per API step to slow down playback if requested
        if eval_delay_s > 0:
            time.sleep(min(eval_delay_s, 5.0))
        
        return jsonify({
            "observations": _serialize_batch_obs(batch_obs),
            "rewards": batch_rewards,
            "dones": batch_dones,
            "infos": batch_infos
        })

    return app


def set_eval_delay(delay_seconds: float):
    global DEFAULT_EVAL_DELAY_S
    DEFAULT_EVAL_DELAY_S = max(0.0, float(delay_seconds))

def init_environments(num_envs, **kwargs):
    print(f"--- Initializing {num_envs} environment(s) with Frame Skipping ({FRAME_SKIPS} skips) ---")
    for i in range(num_envs):
        environments[i] = Game(**kwargs)
        env_locks[i] = threading.Lock()

def run_server(debug=False, quiet=False):
    app = create_app()
    if quiet:
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
    app.run(debug=debug, use_reloader=False)
