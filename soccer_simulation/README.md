# Soccer Simulation for Multi-Agent Reinforcement Learning

This project provides a 2v2 soccer simulation environment designed for multi-agent reinforcement learning (MARL). It features a client-server architecture that allows AI agents (or a test script) to control players by sending actions and receiving observations.

## Core Architecture

The simulation is built on a decoupled, non-blocking client-server model to ensure smooth physics and a responsive API.

*   **Continuous Physics, Burst Simulation**: The simulation runs on a Python Flask server. For each parallel environment, a dedicated background thread manages the physics. This thread waits idly until it's triggered by an API call. When triggered, it runs the simulation for a fixed number of ticks (e.g., 8), applying the client's action continuously throughout the burst. This gives every action a clear, finite duration of effect.

*   **Non-Blocking API**: The `/step` endpoint is non-blocking. When a client sends an action, the server immediately returns the observation from the *previous* simulation tick and then signals the appropriate background thread to start its next simulation burst. This allows the client to begin calculating its next move while the consequences of its last action are being simulated on the server.

## File Structure

-   `run.py`
    -   The main entry point to launch the simulation server.
    -   Handles command-line arguments like `--headless` for faster training and `--num-envs` for parallel simulations.

-   `config.json`
    -   A central configuration file for tuning the simulation without changing the code.
    -   Contains physics properties (mass, friction/damping), reward values, and simulation settings.

-   `test_api.py`
    -   An example client that connects to the server and sends random actions.
    -   Useful for visualizing the simulation and testing server functionality.

-   `soccer_simulation/`
    -   **`api/server.py`**: The core Flask server. It defines the API endpoints (`/step`, `/reset_all`) and manages the background physics loops for each environment.
    -   **`game/game.py`**: Contains the main game logic. It manages the agents and the ball, calculates rewards, and handles the `step()` function that advances the physics engine.
    -   **`game/entities.py`**: Defines the `Agent` and `Ball` classes. Crucially, this is where the custom physics logic, including the damping/friction for linear and angular velocity, is implemented.

## How to Run

1.  **Start the Server**:
    ```bash
    python run.py
    ```
    -   For training, run in headless mode for maximum speed: `python run.py --headless`
    -   To run multiple environments in parallel (requires headless mode): `python run.py --headless --num-envs 8`

2.  **Run the Test Client**:
    -   In a separate terminal, start the test client:
    ```bash
    python test_api.py
    ```
    -   If the server is running multiple environments, the client must match: `python test_api.py --num-envs 8`
