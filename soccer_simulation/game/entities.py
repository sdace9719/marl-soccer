import pygame
import pymunk

from game.constants import (AGENT_SIZE, BALL_CATEGORY, AGENT_CATEGORY,
                        WALL_CATEGORY, GOAL_WALL_CATEGORY, BALL_RADIUS,
                        SCREEN_HEIGHT)

class Agent:
    def __init__(self, space, x, y, color, physics_config, angle=0):
        self.id = None # Will be set by the Game class
        self.body = pymunk.Body(physics_config["agent_mass"], 100)
        self.body.position = x, y
        self.body.angle = angle
        
        # Custom velocity function to apply strong damping
        damping = physics_config["agent_friction"]
        max_velocity = physics_config["max_velocity"]
        _default_velocity_func = self.body.velocity_func
        def custom_velocity_func(body, gravity, space_damping, dt):
            # First, call the default Pymunk velocity update
            _default_velocity_func(body, gravity, space_damping, dt)
            # Then, apply our strong custom damping as a per-step multiplier
            body.velocity *= damping
            body.angular_velocity *= damping
            # And finally, cap the max velocity
            if body.velocity.length > max_velocity:
                body.velocity = body.velocity.normalized() * max_velocity
        self.body.velocity_func = custom_velocity_func

        self.shape = pymunk.Poly.create_box(self.body, (AGENT_SIZE, AGENT_SIZE))
        self.shape.elasticity = 0.2
        self.shape.friction = 0.8
        self.shape.filter = pymunk.ShapeFilter(categories=AGENT_CATEGORY, mask=BALL_CATEGORY | AGENT_CATEGORY | WALL_CATEGORY | GOAL_WALL_CATEGORY)
        self.color = color
        space.add(self.body, self.shape)

    def draw(self, screen):
        # Draw the agent's body
        verts = self.shape.get_vertices()
        world_verts = [self.body.local_to_world(v) for v in verts]
        pygame_verts = [(v.x, SCREEN_HEIGHT - v.y) for v in world_verts]
        pygame.draw.polygon(screen, self.color, pygame_verts)

        # Draw the orientation marker as a triangle
        p1_local = pymunk.Vec2d(AGENT_SIZE * 0.5, 0)
        p2_local = pymunk.Vec2d(AGENT_SIZE * 0.25, -AGENT_SIZE * 0.25)
        p3_local = pymunk.Vec2d(AGENT_SIZE * 0.25, AGENT_SIZE * 0.25)

        p1_world = self.body.local_to_world(p1_local)
        p2_world = self.body.local_to_world(p2_local)
        p3_world = self.body.local_to_world(p3_local)

        p1_pygame = (p1_world.x, SCREEN_HEIGHT - p1_world.y)
        p2_pygame = (p2_world.x, SCREEN_HEIGHT - p2_world.y)
        p3_pygame = (p3_world.x, SCREEN_HEIGHT - p3_world.y)

        pygame.draw.polygon(screen, (255, 255, 0), [p1_pygame, p2_pygame, p3_pygame])


class Ball:
    def __init__(self, space, x, y, physics_config):
        self.body = pymunk.Body(physics_config["ball_mass"], 10)
        self.body.position = x, y

        # Custom velocity function to apply moderate damping
        damping = physics_config["ball_friction"]
        max_velocity = physics_config["max_velocity"]
        _default_velocity_func = self.body.velocity_func
        def custom_velocity_func(body, gravity, space_damping, dt):
            # First, call the default Pymunk velocity update
            _default_velocity_func(body, gravity, space_damping, dt)
            # Then, apply our custom damping as a per-step multiplier
            body.velocity *= damping
            # And finally, cap the max velocity
            if body.velocity.length > max_velocity:
                body.velocity = body.velocity.normalized() * max_velocity
        self.body.velocity_func = custom_velocity_func
        
        self.shape = pymunk.Circle(self.body, BALL_RADIUS)
        self.shape.elasticity = 0.95
        self.shape.friction = 0.2
        self.shape.collision_type = 1
        self.shape.filter = pymunk.ShapeFilter(categories=BALL_CATEGORY, mask=AGENT_CATEGORY | WALL_CATEGORY)
        space.add(self.body, self.shape)

    def draw(self, screen):
        x, y = self.body.position
        pygame.draw.circle(screen, (255, 255, 255), (int(x), SCREEN_HEIGHT - int(y)), BALL_RADIUS)
