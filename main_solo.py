from snake_game import SnakeGame
import pygame
import numpy as np
import matplotlib.pyplot as plt
import time


def run(game):
    """Main game loop."""
    
    running = True
    key = pygame.K_RIGHT
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    key = event.key
        
        if game.update(key):
            print(f"Game Over! Your final score was: {game.score}")
            pygame.time.wait(2000)  # Wait for 2 seconds before resetting
            game.reset()
            key = pygame.K_RIGHT
        
        game.draw()
        game.clock.tick(10)  # Controls the speed of the game

if __name__ == '__main__':
    game = SnakeGame()
    run(game)