import pygame
import pymunk
import random
import math
import numpy as np

from .constants import *
from .entities import Agent, Ball

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
            
            # Self State
            self_state = np.array(agent.body.velocity)

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
        # Returns a list of rewards for blue agents only
        all_rewards = {agent.id: 0.0 for agent in self.agents}
        ball_pos = self.ball.body.position

        # 1. Ball Proximity & 2. Kicking/Possession Reward
        agent_distances_to_ball = {agent.id: agent.body.position.get_distance(ball_pos) for agent in self.agents}
        closest_agent_id = min(agent_distances_to_ball, key=agent_distances_to_ball.get)
        
        for agent in self.agents:
            all_rewards[agent.id] += (self._prev_agent_dist_to_ball[agent.id] - agent_distances_to_ball[agent.id]) * self.config["rewards"]["ball_proximity_multiplier"]
            if agent.id == closest_agent_id:
                all_rewards[agent.id] += self.config["rewards"]["kick_possession_reward"]

        # 3. Moving Ball Towards Goal
        ball_dist_to_blue_goal = np.linalg.norm(ball_pos - self.goal_positions["blue_goal"])
        ball_dist_to_red_goal = np.linalg.norm(ball_pos - self.goal_positions["red_goal"])
        
        red_team_reward = (self._prev_ball_dist_to_blue_goal - ball_dist_to_blue_goal) * self.config["rewards"]["move_ball_to_goal_multiplier"]
        blue_team_reward = (self._prev_ball_dist_to_red_goal - ball_dist_to_red_goal) * self.config["rewards"]["move_ball_to_goal_multiplier"]
        
        for agent in self.agents:
            if agent.color == BLUE_TEAM_COLOR: all_rewards[agent.id] += blue_team_reward
            else: all_rewards[agent.id] += red_team_reward

        # 4. Episode End Rewards (NOT SCALED)
        if goal_info["scored"]:
            for agent in self.agents:
                if agent.color == goal_info["scoring_team_color"]:
                    all_rewards[agent.id] += self.config["rewards"]["goal_scored_reward"]
                    all_rewards[agent.id] += self.config["rewards"]["win_reward"]
                else:
                    all_rewards[agent.id] += self.config["rewards"]["goal_conceded_penalty"]
                    all_rewards[agent.id] += self.config["rewards"]["loss_penalty"]
        
        # 5. Alive Penalty
        for agent in self.agents:
            all_rewards[agent.id] += self.config["rewards"]["alive_penalty"]
            
        # Return rewards for blue team only, in a consistent order
        return [all_rewards["agent_0"], all_rewards["agent_1"]]


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
        scoring_team = None
        if ball_pos.x < FIELD_MARGIN and goal_y_bottom < ball_pos.y < goal_y_top:
            print("Goal for Red Team!")
            goal_info = {"scored": True, "scoring_team_color": RED_TEAM_COLOR}
            scoring_team = "red"
        elif ball_pos.x > SCREEN_WIDTH - FIELD_MARGIN and goal_y_bottom < ball_pos.y < goal_y_top:
            print("Goal for Blue Team!")
            goal_info = {"scored": True, "scoring_team_color": BLUE_TEAM_COLOR}
            scoring_team = "blue"
        
        rewards = self._calculate_rewards(goal_info)
        info = {}
        if scoring_team:
            info["goal_scored_by"] = scoring_team

        # Soft reset on goal, allowing the episode to continue
        if goal_info["scored"]:
            self._reset_positions()

        # Episode terminates only when max_steps is reached
        done = False
        if self.max_steps > 0 and self.steps >= self.max_steps:
            done = True

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
        self.clock.tick(60)
