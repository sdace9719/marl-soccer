import pygame
from typing import Optional

from game.constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    FIELD_COLOR,
    LINE_COLOR,
    GOAL_HEIGHT,
    FIELD_MARGIN,
)


class PygameRenderer:
    def __init__(self, window_title: str = "Soccer Simulation"):
        pygame.init()
        self.screen: Optional[pygame.Surface] = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(window_title)
        self.clock = pygame.time.Clock()

    def _draw_field(self):
        self.screen.fill(FIELD_COLOR)
        pygame.draw.line(self.screen, LINE_COLOR, (SCREEN_WIDTH/2, FIELD_MARGIN), (SCREEN_WIDTH/2, SCREEN_HEIGHT - FIELD_MARGIN), 2)
        pygame.draw.circle(self.screen, LINE_COLOR, (SCREEN_WIDTH/2, SCREEN_HEIGHT/2), 70, 2)
        pygame.draw.rect(self.screen, LINE_COLOR, (FIELD_MARGIN, SCREEN_HEIGHT/2 - 150, 120, 300), 2)
        pygame.draw.rect(self.screen, LINE_COLOR, (SCREEN_WIDTH - FIELD_MARGIN - 120, SCREEN_HEIGHT/2 - 150, 120, 300), 2)
        pygame.draw.rect(self.screen, LINE_COLOR, (FIELD_MARGIN - 10, SCREEN_HEIGHT/2 - GOAL_HEIGHT/2, 10, GOAL_HEIGHT), 0)
        pygame.draw.rect(self.screen, LINE_COLOR, (SCREEN_WIDTH - FIELD_MARGIN, SCREEN_HEIGHT/2 - GOAL_HEIGHT/2, 10, GOAL_HEIGHT), 0)

    def draw(self, game):
        # Basic event pump to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                return

        self._draw_field()
        for agent in game.agents:
            agent.draw(self.screen)
        game.ball.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)


