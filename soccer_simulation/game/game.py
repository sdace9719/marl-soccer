import pygame
import pymunk
import random
import math
import numpy as np

from game.constants import *
from game.entities import Agent, Ball

class Game:
    def __init__(self, config, headless=False):
        self.config = config
        self.headless = headless
        # Positioning behavior
        self._use_fixed_positions = False
        self._use_full_random_positions = False
        self._rng = np.random.default_rng()
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

    def reset(self, use_fixed_positions: bool = False, use_full_random_positions: bool = False, seed: int | None = None):
        self.steps = 0
        # Configure positioning mode and RNG
        self._use_fixed_positions = bool(use_fixed_positions)
        self._use_full_random_positions = bool(use_full_random_positions)
        if seed is not None:
            try:
                self._rng = np.random.default_rng(int(seed))
            except Exception:
                self._rng = np.random.default_rng()
        # Reset score for the episode
        self.score = {"blue": 0, "red": 0}
        for agent in self.agents:
            self.space.remove(agent.body, agent.shape)
        if self.ball:
            self.space.remove(self.ball.body, self.ball.shape)

        self.agents = []
        # Create agents (initial positions will be adjusted below)
        self.agents.append(Agent(self.space, SCREEN_WIDTH * 0.25, SCREEN_HEIGHT * 0.33, BLUE_TEAM_COLOR, self.config["physics"]))
        self.agents.append(Agent(self.space, SCREEN_WIDTH * 0.25, SCREEN_HEIGHT * 0.66, BLUE_TEAM_COLOR, self.config["physics"]))
        self.agents.append(Agent(self.space, SCREEN_WIDTH * 0.75, SCREEN_HEIGHT * 0.33, RED_TEAM_COLOR, self.config["physics"], angle=math.pi))
        self.agents.append(Agent(self.space, SCREEN_WIDTH * 0.75, SCREEN_HEIGHT * 0.66, RED_TEAM_COLOR, self.config["physics"], angle=math.pi))

        # Assign agent IDs
        self.agent_ids = [f"agent_{i}" for i in range(len(self.agents))]
        for i, agent in enumerate(self.agents):
            agent.id = self.agent_ids[i]

        # Ball (initial position will be adjusted below)
        self.ball = Ball(self.space, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, self.config["physics"])
        
        # Apply positioning mode
        if self._use_fixed_positions:
            self._apply_fixed_positions()
        elif self._use_full_random_positions:
            self._apply_full_random_positions()
        else:
            self._apply_random_positions()
        
        self._update_reward_state()
        
        return self._get_observations()

    def _reset_positions(self):
        """Resets the positions and velocities of agents and ball without ending the episode."""
        if self._use_fixed_positions:
            self._apply_fixed_positions()
        elif self._use_full_random_positions:
            self._apply_full_random_positions()
        else:
            self._apply_random_positions()

    def _apply_fixed_positions(self):
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

    def _apply_random_positions(self):
        margin = float(FIELD_MARGIN + 20)
        left_x_min = margin
        left_x_max = float(SCREEN_WIDTH / 2 - 20)
        right_x_min = float(SCREEN_WIDTH / 2 + 20)
        right_x_max = float(SCREEN_WIDTH - FIELD_MARGIN - 20)
        y_min = margin
        y_max = float(SCREEN_HEIGHT - FIELD_MARGIN - 20)

        # Blue agents on left half
        bx1 = float(self._rng.uniform(left_x_min, left_x_max))
        by1 = float(self._rng.uniform(y_min, y_max))
        bx2 = float(self._rng.uniform(left_x_min, left_x_max))
        by2 = float(self._rng.uniform(y_min, y_max))
        # Red agents on right half
        rx1 = float(self._rng.uniform(right_x_min, right_x_max))
        ry1 = float(self._rng.uniform(y_min, y_max))
        rx2 = float(self._rng.uniform(right_x_min, right_x_max))
        ry2 = float(self._rng.uniform(y_min, y_max))
        # Ball around center with some spread
        cx = float(SCREEN_WIDTH / 2 + self._rng.uniform(-40.0, 40.0))
        cy = float(SCREEN_HEIGHT / 2 + self._rng.uniform(-40.0, 40.0))

        # Apply positions and zero velocities
        self.agents[0].body.position = bx1, by1
        self.agents[1].body.position = bx2, by2
        self.agents[2].body.position = rx1, ry1
        self.agents[3].body.position = rx2, ry2

        for idx, agent in enumerate(self.agents):
            agent.body.velocity = (0, 0)
            agent.body.angular_velocity = 0
            # Set facing directions as in fixed start
            agent.body.angle = 0 if idx < 2 else math.pi

        self.ball.body.position = cx, cy
        self.ball.body.velocity = (0, 0)

    def _apply_full_random_positions(self):
        margin = float(FIELD_MARGIN + 20)
        x_min = margin
        x_max = float(SCREEN_WIDTH - FIELD_MARGIN - 20)
        y_min = margin
        y_max = float(SCREEN_HEIGHT - FIELD_MARGIN - 20)

        # Corner bias: 75% chance both blue agents spawn near corners (any of the four)
        blue_corners = self._rng.uniform(0.0, 1.0) < 0.75

        def sample_corner(corner_idx: int):
            # corner_idx: 0=top-left, 1=bottom-left, 2=top-right, 3=bottom-right
            pad = 8.0  # smaller offset from the walls for tighter corner spawns
            left = corner_idx in (0, 1)
            top = corner_idx in (0, 2)
            cx = float(FIELD_MARGIN + pad) if left else float(SCREEN_WIDTH - FIELD_MARGIN - pad)
            cy = float(SCREEN_HEIGHT - FIELD_MARGIN - pad) if top else float(FIELD_MARGIN + pad)
            # jitter within a small box
            jx = float(self._rng.uniform(-5.0, 5.0))
            jy = float(self._rng.uniform(-5.0, 5.0))
            return cx + jx, cy + jy

        # Blue agents
        if blue_corners:
            # Pick two corners (could be same side or different) uniformly from 4
            c1 = int(self._rng.integers(0, 4))
            c2 = int(self._rng.integers(0, 4))
            bx1, by1 = sample_corner(c1)
            bx2, by2 = sample_corner(c2)
        else:
            bx1 = float(self._rng.uniform(x_min, x_max))
            by1 = float(self._rng.uniform(y_min, y_max))
            bx2 = float(self._rng.uniform(x_min, x_max))
            by2 = float(self._rng.uniform(y_min, y_max))

        # Red agents anywhere
        rx1 = float(self._rng.uniform(x_min, x_max))
        ry1 = float(self._rng.uniform(y_min, y_max))
        rx2 = float(self._rng.uniform(x_min, x_max))
        ry2 = float(self._rng.uniform(y_min, y_max))

        # Ball anywhere on field
        cx = float(self._rng.uniform(x_min, x_max))
        cy = float(self._rng.uniform(y_min, y_max))

        # Apply positions and zero velocities
        self.agents[0].body.position = bx1, by1
        self.agents[1].body.position = bx2, by2
        self.agents[2].body.position = rx1, ry1
        self.agents[3].body.position = rx2, ry2

        for idx, agent in enumerate(self.agents):
            agent.body.velocity = (0, 0)
            agent.body.angular_velocity = 0
            agent.body.angle = 0 if idx < 2 else math.pi

        self.ball.body.position = cx, cy
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
        physics_cfg = self.config.get("physics", {})
        max_velocity = float(physics_cfg.get("max_velocity", 400.0))
        # Estimate max angular velocity; allow override via config
        max_ang_vel = float(physics_cfg.get("max_angular_velocity", physics_cfg.get("action_torque_max", 100000.0) / 100.0))
        field_diag = float(np.hypot(SCREEN_WIDTH, SCREEN_HEIGHT))
        for i, agent in enumerate(self.agents):
            agent_pos = np.array(agent.body.position)

            # Self State: velocity normalized by max_velocity, angle normalized by pi (wrapped to [-pi, pi])
            vel = np.array(agent.body.velocity, dtype=np.float32) / max(max_velocity, 1e-6)
            # wrap angle to [-pi, pi]
            ang = float(agent.body.angle)
            ang_wrapped = math.atan2(math.sin(ang), math.cos(ang))
            ang_norm = np.array([ang_wrapped / math.pi], dtype=np.float32)
            ang_vel_norm = np.array([float(agent.body.angular_velocity) / max(max_ang_vel, 1e-6)], dtype=np.float32)
            self_state = np.concatenate([vel, ang_norm, ang_vel_norm])

            # Helper: convert vector to (unit_x, unit_y, magnitude)
            def vec_to_unit_mag(vec2):
                mag = float(np.linalg.norm(vec2))
                if mag > 1e-8:
                    unit = (vec2 / mag).astype(np.float32)
                else:
                    unit = np.array([0.0, 0.0], dtype=np.float32)
                    mag = 0.0
                # normalize magnitude by maximum possible distance (field diagonal)
                mag_norm = mag / max(field_diag, 1e-6)
                return unit, mag_norm

            # Teammate Info
            teammate = self.agents[1] if i == 0 else self.agents[0] if i == 1 else self.agents[3] if i == 2 else self.agents[2]
            teammate_vec = np.array(teammate.body.position) - agent_pos
            teammate_unit, teammate_mag = vec_to_unit_mag(teammate_vec)

            # Opponent Info
            opponents = [self.agents[j] for j in range(4) if self.agents[j].color != agent.color]
            opp1_vec = np.array(opponents[0].body.position) - agent_pos
            opp2_vec = np.array(opponents[1].body.position) - agent_pos
            opp1_unit, opp1_mag = vec_to_unit_mag(opp1_vec)
            opp2_unit, opp2_mag = vec_to_unit_mag(opp2_vec)

            # Ball Info
            ball_vec = np.array(self.ball.body.position) - agent_pos
            ball_unit, ball_mag = vec_to_unit_mag(ball_vec)

            # Goal Info
            own_goal, opp_goal = (("blue_goal", "red_goal") if agent.color == BLUE_TEAM_COLOR else ("red_goal", "blue_goal"))
            own_goal_vec = self.goal_positions[own_goal] - agent_pos
            opp_goal_vec = self.goal_positions[opp_goal] - agent_pos
            own_unit, own_mag = vec_to_unit_mag(own_goal_vec)
            opp_unit, opp_mag = vec_to_unit_mag(opp_goal_vec)
            
            observations.append(np.concatenate([
                self_state,                        # [vx, vy, angle] -> 3
                teammate_unit, np.array([teammate_mag]),  # 2 + 1
                opp1_unit,     np.array([opp1_mag]),      # 2 + 1
                opp2_unit,     np.array([opp2_mag]),      # 2 + 1
                ball_unit,     np.array([ball_mag]),      # 2 + 1
                own_unit,      np.array([own_mag]),       # 2 + 1
                opp_unit,      np.array([opp_mag])        # 2 + 1
            ]))
        return observations

    def _calculate_rewards(self, goal_info):
        # Returns rewards for blue agents only (agent_0, agent_1)
        ball_pos = self.ball.body.position
        blue_rewards = {"agent_0": 0.0, "agent_1": 0.0}

        # 1) Ball proximity shaping (team-level, equal for both blue agents)
        prox_mult = float(self.config["rewards"].get("ball_proximity_multiplier", 0.0))
        if prox_mult != 0.0:
            # Improvement towards the ball since last tick for both blue agents
            agent0, agent1 = self.agents[0], self.agents[1]
            prev_d0 = float(self._prev_agent_dist_to_ball.get(agent0.id, 0.0))
            prev_d1 = float(self._prev_agent_dist_to_ball.get(agent1.id, 0.0))
            curr_d0 = float(agent0.body.position.get_distance(ball_pos))
            curr_d1 = float(agent1.body.position.get_distance(ball_pos))
            team_prox_improvement = (prev_d0 - curr_d0) + (prev_d1 - curr_d1)
            team_prox_reward = prox_mult * team_prox_improvement
            blue_rewards["agent_0"] += team_prox_reward
            blue_rewards["agent_1"] += team_prox_reward

        # 2) Moving Ball Towards Opponent Goal (blue perspective)
        ball_dist_to_red_goal = np.linalg.norm(ball_pos - self.goal_positions["red_goal"])
        ball_to_goal_improvement = self._prev_ball_dist_to_red_goal - ball_dist_to_red_goal
        blue_team_reward = ball_to_goal_improvement * self.config["rewards"]["move_ball_to_goal_multiplier"]
        blue_rewards["agent_0"] += blue_team_reward
        blue_rewards["agent_1"] += blue_team_reward

        # 3) Kick possession reward (disabled for now)
        # kick_reward = float(self.config["rewards"].get("kick_possession_reward", 0.0))
        # if kick_reward != 0.0:
        #     # Heuristic: if any blue agent is within a contact threshold to the ball
        #     agent0, agent1 = self.agents[0], self.agents[1]
        #     d0 = float(agent0.body.position.get_distance(ball_pos))
        #     d1 = float(agent1.body.position.get_distance(ball_pos))
        #     contact_threshold = float(AGENT_SIZE * 0.6 + BALL_RADIUS * 1.2)
        #     if (d0 <= contact_threshold) or (d1 <= contact_threshold):
        #         blue_rewards["agent_0"] += kick_reward
        #         blue_rewards["agent_1"] += kick_reward

        # 4. Episode End Rewards (blue perspective only)
        if goal_info["scored"]:
            if goal_info["scoring_team_color"] == BLUE_TEAM_COLOR:
                blue_rewards["agent_0"] += self.config["rewards"]["goal_scored_reward"]
                blue_rewards["agent_1"] += self.config["rewards"]["goal_scored_reward"]
            else:
                blue_rewards["agent_0"] -= self.config["rewards"]["goal_conceded_penalty"]
                blue_rewards["agent_1"] -= self.config["rewards"]["goal_conceded_penalty"]

        # 5. Alive Penalty (blue only)
        blue_rewards["agent_0"] -= self.config["rewards"]["alive_penalty"]
        blue_rewards["agent_1"] -= self.config["rewards"]["alive_penalty"]

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
            goal_info = {"scored": True, "scoring_team_color": RED_TEAM_COLOR}
            self.score["red"] += 1
        elif ball_pos.x > SCREEN_WIDTH - FIELD_MARGIN and goal_y_bottom < ball_pos.y < goal_y_top:
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
