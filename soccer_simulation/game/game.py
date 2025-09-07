import pygame
import pymunk
import random
import math
import numpy as np

from soccer_simulation.game.constants import *
from soccer_simulation.game.entities import Agent, Ball

class Game:
    def __init__(self, config, headless=False):
        self.config = config
        self.headless = headless
        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Soccer Simulation")
            self.clock = pygame.time.Clock()

        self.space = pymunk.Space()
        self.space.gravity = (0, 0)

        self.max_steps = self.config["simulation"]["max_steps"]
        self.steps = 0
        
        self.agents = []
        self.agent_ids = []
        self.ball = None
        self.goal_positions = {
            "blue_goal": np.array([FIELD_MARGIN, SCREEN_HEIGHT / 2]),
            "red_goal": np.array([SCREEN_WIDTH - FIELD_MARGIN, SCREEN_HEIGHT / 2])
        }
        
        # State for reward calculation
        self._prev_agent_dist_to_ball = {}
        self._prev_ball_dist_to_blue_goal = 0
        self._prev_ball_dist_to_red_goal = 0
        
        self.setup_field()

    def setup_field(self):
        goal_y_top = SCREEN_HEIGHT/2 + GOAL_HEIGHT/2
        goal_y_bottom = SCREEN_HEIGHT/2 - GOAL_HEIGHT/2

        # Walls
        static_lines = [
            pymunk.Segment(self.space.static_body, (FIELD_MARGIN, FIELD_MARGIN), (SCREEN_WIDTH - FIELD_MARGIN, FIELD_MARGIN), 2),
            pymunk.Segment(self.space.static_body, (FIELD_MARGIN, SCREEN_HEIGHT - FIELD_MARGIN), (SCREEN_WIDTH - FIELD_MARGIN, SCREEN_HEIGHT - FIELD_MARGIN), 2),
            pymunk.Segment(self.space.static_body, (FIELD_MARGIN, FIELD_MARGIN), (FIELD_MARGIN, goal_y_bottom), 2),
            pymunk.Segment(self.space.static_body, (FIELD_MARGIN, goal_y_top), (FIELD_MARGIN, SCREEN_HEIGHT - FIELD_MARGIN), 2),
            pymunk.Segment(self.space.static_body, (SCREEN_WIDTH - FIELD_MARGIN, FIELD_MARGIN), (SCREEN_WIDTH - FIELD_MARGIN, goal_y_bottom), 2),
            pymunk.Segment(self.space.static_body, (SCREEN_WIDTH - FIELD_MARGIN, goal_y_top), (SCREEN_WIDTH - FIELD_MARGIN, SCREEN_HEIGHT - FIELD_MARGIN), 2),
        ]
        for line in static_lines:
            line.elasticity = 0.95
            line.friction = 0.2
            line.filter = pymunk.ShapeFilter(categories=WALL_CATEGORY, mask=AGENT_CATEGORY | BALL_CATEGORY)
        self.space.add(*static_lines)

        # Invisible walls for agents in goals
        goal_lines = [
            pymunk.Segment(self.space.static_body, (FIELD_MARGIN, goal_y_bottom), (FIELD_MARGIN, goal_y_top), 1),
            pymunk.Segment(self.space.static_body, (SCREEN_WIDTH - FIELD_MARGIN, goal_y_bottom), (SCREEN_WIDTH - FIELD_MARGIN, goal_y_top), 1)
        ]
        for line in goal_lines:
            line.elasticity = 0.95
            line.filter = pymunk.ShapeFilter(categories=GOAL_WALL_CATEGORY, mask=AGENT_CATEGORY)
        self.space.add(*goal_lines)
        
        self.reset()

    def reset(self):
        self.steps = 0
        # Reset score for the episode
        self.score = {"blue": 0, "red": 0}
        for agent in self.agents:
            self.space.remove(agent.body, agent.shape)
        if self.ball:
            self.space.remove(self.ball.body, self.ball.shape)

        self.agents = []
        # Team Blue
        self.agents.append(Agent(self.space, SCREEN_WIDTH * 0.25, SCREEN_HEIGHT * 0.33, BLUE_TEAM_COLOR, self.config["physics"]))
        self.agents.append(Agent(self.space, SCREEN_WIDTH * 0.25, SCREEN_HEIGHT * 0.66, BLUE_TEAM_COLOR, self.config["physics"]))
        
        # Team Red
        self.agents.append(Agent(self.space, SCREEN_WIDTH * 0.75, SCREEN_HEIGHT * 0.33, RED_TEAM_COLOR, self.config["physics"], angle=math.pi))
        self.agents.append(Agent(self.space, SCREEN_WIDTH * 0.75, SCREEN_HEIGHT * 0.66, RED_TEAM_COLOR, self.config["physics"], angle=math.pi))

        # Assign agent IDs
        self.agent_ids = [f"agent_{i}" for i in range(len(self.agents))]
        for i, agent in enumerate(self.agents):
            agent.id = self.agent_ids[i]

        # Ball
        self.ball = Ball(self.space, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, self.config["physics"])
        
        self._update_reward_state()
        
        return self._get_observations()

    def _reset_positions(self):
        """Resets the positions and velocities of agents and ball without ending the episode."""
        # Team Blue
        self.agents[0].body.position = SCREEN_WIDTH * 0.25, SCREEN_HEIGHT * 0.33
        self.agents[1].body.position = SCREEN_WIDTH * 0.25, SCREEN_HEIGHT * 0.66
        self.agents[0].body.velocity = (0, 0)
        self.agents[1].body.velocity = (0, 0)
        self.agents[0].body.angle = 0
        self.agents[1].body.angle = 0
        self.agents[0].body.angular_velocity = 0
        self.agents[1].body.angular_velocity = 0

        # Team Red
        self.agents[2].body.position = SCREEN_WIDTH * 0.75, SCREEN_HEIGHT * 0.33
        self.agents[3].body.position = SCREEN_WIDTH * 0.75, SCREEN_HEIGHT * 0.66
        self.agents[2].body.velocity = (0, 0)
        self.agents[3].body.velocity = (0, 0)
        self.agents[2].body.angle = math.pi
        self.agents[3].body.angle = math.pi
        self.agents[2].body.angular_velocity = 0
        self.agents[3].body.angular_velocity = 0

        # Ball
        self.ball.body.position = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
        self.ball.body.velocity = (0, 0)

    def _update_reward_state(self):
        ball_pos = self.ball.body.position
        for agent in self.agents:
            self._prev_agent_dist_to_ball[agent.id] = agent.body.position.get_distance(ball_pos)
        self._prev_ball_dist_to_blue_goal = np.linalg.norm(ball_pos - self.goal_positions["blue_goal"])
        self._prev_ball_dist_to_red_goal = np.linalg.norm(ball_pos - self.goal_positions["red_goal"])

    def _get_observations(self):
        # Returns a list of observations, ordered by agent (blue team first)
        observations = []
        for i, agent in enumerate(self.agents):
            agent_pos = np.array(agent.body.position)
            
            # Self State: position, velocity, angle
            self_state = np.concatenate([
                agent_pos,
                np.array(agent.body.velocity),
                np.array([agent.body.angle])
            ])

            # Teammate Info
            teammate = self.agents[1] if i == 0 else self.agents[0] if i == 1 else self.agents[3] if i == 2 else self.agents[2]
            teammate_vec = np.array(teammate.body.position) - agent_pos

            # Opponent Info
            opponents = [self.agents[j] for j in range(4) if self.agents[j].color != agent.color]
            opp1_vec = np.array(opponents[0].body.position) - agent_pos
            opp2_vec = np.array(opponents[1].body.position) - agent_pos

            # Ball Info
            ball_vec = np.array(self.ball.body.position) - agent_pos

            # Goal Info
            own_goal, opp_goal = (("blue_goal", "red_goal") if agent.color == BLUE_TEAM_COLOR else ("red_goal", "blue_goal"))
            own_goal_vec = self.goal_positions[own_goal] - agent_pos
            opp_goal_vec = self.goal_positions[opp_goal] - agent_pos
            
            observations.append(np.concatenate([
                self_state, teammate_vec, opp1_vec, opp2_vec, ball_vec, own_goal_vec, opp_goal_vec
            ]))
        return observations

    def _calculate_rewards(self, goal_info):
        # Returns rewards for blue agents only (agent_0, agent_1)
        ball_pos = self.ball.body.position
        blue_rewards = {"agent_0": 0.0, "agent_1": 0.0}

        # 1. Ball Proximity & 2. Kicking/Possession Reward (only accrue to blue agents)
        agent_distances_to_ball = {agent.id: agent.body.position.get_distance(ball_pos) for agent in self.agents}
        closest_agent_id = min(agent_distances_to_ball, key=agent_distances_to_ball.get)

        for agent in self.agents[:2]:  # blue team: indices 0 and 1
            agent_id = agent.id
            blue_rewards[agent_id] += (
                self._prev_agent_dist_to_ball[agent_id] - agent_distances_to_ball[agent_id]
            ) * self.config["rewards"]["ball_proximity_multiplier"]

        if closest_agent_id in ("agent_0", "agent_1"):
            blue_rewards[closest_agent_id] += self.config["rewards"]["kick_possession_reward"]

        # 3. Moving Ball Towards Opponent Goal (blue perspective)
        ball_dist_to_red_goal = np.linalg.norm(ball_pos - self.goal_positions["red_goal"])
        blue_team_reward = (
            self._prev_ball_dist_to_red_goal - ball_dist_to_red_goal
        ) * self.config["rewards"]["move_ball_to_goal_multiplier"]
        blue_rewards["agent_0"] += blue_team_reward
        blue_rewards["agent_1"] += blue_team_reward

        # 4. Episode End Rewards (blue perspective only)
        if goal_info["scored"]:
            if goal_info["scoring_team_color"] == BLUE_TEAM_COLOR:
                blue_rewards["agent_0"] += self.config["rewards"]["goal_scored_reward"]
                blue_rewards["agent_1"] += self.config["rewards"]["goal_scored_reward"]
            else:
                blue_rewards["agent_0"] += self.config["rewards"]["goal_conceded_penalty"]
                blue_rewards["agent_1"] += self.config["rewards"]["goal_conceded_penalty"]

        # 5. Alive Penalty (blue only)
        blue_rewards["agent_0"] += self.config["rewards"]["alive_penalty"]
        blue_rewards["agent_1"] += self.config["rewards"]["alive_penalty"]

        return [blue_rewards["agent_0"], blue_rewards["agent_1"]]


    def step(self, actions=None):
        self._update_reward_state()
        self.steps += 1
        
        # Reset forces from previous step
        for agent in self.agents:
            agent.body.force = (0, 0)
            agent.body.torque = 0
        self.ball.body.force = (0, 0)
        self.ball.body.torque = 0
        
        # Apply actions from the client
        if actions:
            for agent in self.agents:
                agent_action = actions.get(agent.id)
                if agent_action:
                    force = (agent_action[0], agent_action[1])
                    torque = agent_action[2]
                    agent.body.apply_force_at_local_point(force, (0, 0))
                    agent.body.torque = torque

        self.space.step(1/60.0)

        # Check for goal
        ball_pos = self.ball.body.position
        goal_y_top = SCREEN_HEIGHT/2 + GOAL_HEIGHT/2
        goal_y_bottom = SCREEN_HEIGHT/2 - GOAL_HEIGHT/2

        goal_info = {"scored": False}
        if ball_pos.x < FIELD_MARGIN and goal_y_bottom < ball_pos.y < goal_y_top:
            print("Goal for Red Team!")
            goal_info = {"scored": True, "scoring_team_color": RED_TEAM_COLOR}
            self.score["red"] += 1
        elif ball_pos.x > SCREEN_WIDTH - FIELD_MARGIN and goal_y_bottom < ball_pos.y < goal_y_top:
            print("Goal for Blue Team!")
            goal_info = {"scored": True, "scoring_team_color": BLUE_TEAM_COLOR}
            self.score["blue"] += 1
        
        rewards = self._calculate_rewards(goal_info)
        info = {"score": {"blue": self.score.get("blue", 0), "red": self.score.get("red", 0)}}
        if goal_info["scored"]:
            scoring_team = "blue" if goal_info["scoring_team_color"] == BLUE_TEAM_COLOR else "red"
            info["goal_scored_by"] = scoring_team

        # Soft reset on goal, allowing the episode to continue
        if goal_info["scored"]:
            self._reset_positions()

        # Episode terminates only when max_steps is reached
        done = False
        if self.max_steps > 0 and self.steps >= self.max_steps:
            done = True
            # Terminal bonus based on score difference (blue perspective)
            score_diff = self.score.get("blue", 0) - self.score.get("red", 0)
            multiplier = float(self.config["rewards"].get("score_difference_multiplier", 5.0))
            terminal_bonus = multiplier * score_diff
            # On the terminal step, only return the terminal bonus as the step reward
            rewards = [terminal_bonus, terminal_bonus]

        observations = self._get_observations()
        
        return observations, rewards, done, info


    def draw_field(self):
        if self.headless: return
        self.screen.fill(FIELD_COLOR)
        pygame.draw.line(self.screen, LINE_COLOR, (SCREEN_WIDTH/2, FIELD_MARGIN), (SCREEN_WIDTH/2, SCREEN_HEIGHT - FIELD_MARGIN), 2)
        pygame.draw.circle(self.screen, LINE_COLOR, (SCREEN_WIDTH/2, SCREEN_HEIGHT/2), 70, 2)
        pygame.draw.rect(self.screen, LINE_COLOR, (FIELD_MARGIN, SCREEN_HEIGHT/2 - 150, 120, 300), 2)
        pygame.draw.rect(self.screen, LINE_COLOR, (SCREEN_WIDTH - FIELD_MARGIN - 120, SCREEN_HEIGHT/2 - 150, 120, 300), 2)
        pygame.draw.rect(self.screen, LINE_COLOR, (FIELD_MARGIN - 10, SCREEN_HEIGHT/2 - GOAL_HEIGHT/2, 10, GOAL_HEIGHT), 0)
        pygame.draw.rect(self.screen, LINE_COLOR, (SCREEN_WIDTH - FIELD_MARGIN, SCREEN_HEIGHT/2 - GOAL_HEIGHT/2, 10, GOAL_HEIGHT), 0)

    def draw(self):
        if self.headless: return
        self.draw_field()
        for agent in self.agents:
            agent.draw(self.screen)
        self.ball.draw(self.screen)
        pygame.display.flip()
