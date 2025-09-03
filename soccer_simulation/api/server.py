from flask import Flask, jsonify, request
import threading
import numpy as np
import logging
import time
import atexit

from ..game.game import Game

# --- Constants ---
NUM_AGENTS = 4
ACTION_DIM = 3
SIM_TICKS_PER_ACTION = 8 # Number of physics steps to run per action

# --- Globals for Burst Simulation Architecture ---
environments = {}
env_locks = {}
last_actions = {}       # Holds the last action from the client
latest_results = {}     # Stores the most recent results from the physics threads
start_tick_events = {}  # Events to trigger the physics loop
stop_threads = False

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
                latest_results[env_id] = (obs, [0.0]*2, False, {})
                last_actions[env_id] = None
        return jsonify({"observations": _serialize_batch_obs(batch_obs)})

    @app.route('/step', methods=['POST'])
    def step():
        actions_list = request.json.get('actions')
        # --- (Input Validation) ---
        num_envs = len(environments)
        if not isinstance(actions_list, list) or len(actions_list) != num_envs:
            return jsonify({"error": "Invalid 'actions' format."}), 400
        
        # --- Read current state, then trigger next simulation burst ---
        batch_obs = [[] for _ in range(num_envs)]
        batch_rewards = [[] for _ in range(num_envs)]
        batch_dones = [False for _ in range(num_envs)]
        batch_infos = [{} for _ in range(num_envs)]
        for env_id, actions in enumerate(actions_list):
            with env_locks[env_id]:
                # 1. Immediately read the latest available state
                obs, rewards, done, info = latest_results[env_id]
                batch_obs[env_id] = obs
                batch_rewards[env_id] = rewards
                batch_dones[env_id] = done
                batch_infos[env_id] = info

                # 2. Set the next action and trigger the physics loop
                last_actions[env_id] = actions
                start_tick_events[env_id].set()
        
        # 3. Return the state that was just read (non-blocking)
        return jsonify({
            "observations": _serialize_batch_obs(batch_obs),
            "rewards": batch_rewards,
            "dones": batch_dones,
            "infos": batch_infos
        })

    return app

def physics_loop(env_id, step_interval=1/60.0):
    """A background thread that waits for a signal, then runs the simulation for a fixed number of ticks."""
    while not stop_threads:
        # 1. Wait until an action is received from the /step endpoint
        start_tick_events[env_id].wait()
        start_tick_events[env_id].clear()

        # 2. Get the action that triggered this burst
        current_action = None
        with env_locks[env_id]:
            current_action = last_actions.get(env_id)

        # 3. Run the simulation for a fixed number of ticks, applying the same action each time
        for _ in range(SIM_TICKS_PER_ACTION):
            obs, rewards, done, info = environments[env_id].step(current_action)
            
            # Store the latest results after each tick
            with env_locks[env_id]:
                latest_results[env_id] = (obs, rewards, done, info)
            
            if step_interval > 0:
                time.sleep(step_interval)
    
def init_environments(num_envs, **kwargs):
    is_headless = kwargs.get("headless", False)
    step_interval = 0.0 if is_headless else 1/60.0
    print(f"--- Running in {'Headless' if is_headless else 'Graphical'} mode. Physics step interval: {step_interval:.4f}s ---")

    for i in range(num_envs):
        environments[i] = Game(**kwargs)
        env_locks[i] = threading.Lock()
        latest_results[i] = ([], [0.0]*2, False, {})
        last_actions[i] = None
        start_tick_events[i] = threading.Event()

        thread = threading.Thread(target=physics_loop, args=(i,), kwargs={"step_interval": step_interval})
        thread.daemon = True
        thread.start()

def run_server(debug=False, quiet=False):
    app = create_app()
    if quiet:
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
    app.run(debug=debug, use_reloader=False)

def _stop_background_threads():
    global stop_threads
    stop_threads = True
    # Set all events to unblock any waiting threads on shutdown
    for event in start_tick_events.values():
        event.set()

atexit.register(_stop_background_threads)
