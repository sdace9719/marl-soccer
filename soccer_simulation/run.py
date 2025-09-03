import argparse
import threading
import pygame
import logging
import json

from api.server import run_server, init_environments
from game.constants import *

def main():
    parser = argparse.ArgumentParser(description="Soccer Simulation")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without graphics")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments to run (requires headless mode)")
    parser.add_argument("--quiet", action="store_true", help="Disable server logs")
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

    # Start the server in a separate thread
    server_thread = threading.Thread(target=run_server, kwargs={"quiet": args.quiet})
    server_thread.daemon = True
    server_thread.start()
    
    if not args.quiet:
        print(f"Server listening on http://127.0.0.1:5000")
    
    # If in graphical mode, run the rendering loop
    if not args.headless:
        game = environments[0]
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # In graphical mode, we only draw the current state.
            # The simulation is advanced by API calls from a client (e.g., test_api.py).
            game.draw()
        
        pygame.quit()
    else:
        # In headless mode, just keep the main thread alive
        server_thread.join()


if __name__ == "__main__":
    main()
