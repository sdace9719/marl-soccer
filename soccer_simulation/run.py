import os
import sys
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import threading
import pygame
import logging
import json

from soccer_simulation.api.server import run_server, init_environments, environments, set_eval_delay
from soccer_simulation.game.constants import *

def main():
    parser = argparse.ArgumentParser(description="Soccer Simulation")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without graphics")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments to run (requires headless mode)")
    parser.add_argument("--quiet", action="store_true", help="Disable server logs")
    parser.add_argument("--eval", action="store_true", help="Enable evaluation mode (intended for slow visual playback)")
    parser.add_argument("--eval-delay", type=float, default=0.5, help="Evaluation delay in seconds per API step (used when --eval)")
    args = parser.parse_args()

    # Load configuration from file
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.num_envs > 1 and not args.headless:
        raise ValueError("Cannot run multiple environments with graphics enabled. Please use --headless.")

    # Initialize environments
    game_kwargs = {
        "config": config,
        "headless": args.headless
    }
    init_environments(args.num_envs, **game_kwargs)

    # Configure default server-side eval delay if requested
    if args.eval:
        set_eval_delay(args.eval_delay)
        if not args.quiet:
            print(f"Eval mode enabled: server will sleep {args.eval_delay:.2f}s per step")

    # Start the server in a separate thread
    # Note: eval delay is passed by clients with each /step call via 'eval_delay_s'
    server_thread = threading.Thread(target=run_server, kwargs={"quiet": args.quiet})
    server_thread.daemon = True
    server_thread.start()
    
    if not args.quiet:
        print(f"Server listening on http://127.0.0.1:5000")
    
    # If in graphical mode, run the rendering loop
    if not args.headless:
        game = environments[0]
        running = True
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                
                # In graphical mode, we only draw the current state.
                # The simulation is advanced by API calls from a client (e.g., test_api.py).
                game.draw()
        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt, shutting down.")
        finally:
            pygame.quit()
    else:
        # In headless mode, just keep the main thread alive
        server_thread.join()


if __name__ == "__main__":
    main()
