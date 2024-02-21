from snake_game import SnakeGame
import pygame
import numpy as np
from model import DQNModel  # Make sure this is the path to your model file

tickVal = 20
def run(game):

    # Initialize game
    game = SnakeGame()
    clock = pygame.time.Clock()
    stateBib = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]

    """Main game loop."""
    # Model setup
    input_shape = (game.HEIGHTM, game.WIDTHM, 2)  # Input shape for the model
    action_size = 4  # Number of possible actions (up, down)
    model = DQNModel(input_shape, action_size,True) # Initialize the model

    
    # Initial states
    game.update(pygame.K_RIGHT)
    game.draw()
    clock.tick(tickVal)
    prev_state = game.get_game_state()
    game.update(pygame.K_RIGHT)
    game.draw()
    clock.tick(tickVal)
    current_state = game.get_game_state()
    state = np.stack((prev_state, current_state), axis=-1)  # Combine two frames
    
    
    numFrames = 0
    oldGameScore = game.score
    running = True
    key = pygame.K_RIGHT
    
    
    while running:
        numFrames += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Model prediction 
        action = model.predict(state)
        
        # Update game state based on model predictions
        game_over = game.update(stateBib[action])
        
        game.draw()
        game.clock.tick(tickVal)  # Controls the speed of the game
        
        # Prepare for the next state
        prev_state = current_state
        current_state = game.get_game_state()
        state = np.stack((prev_state, current_state), axis=-1)
        
        if oldGameScore < game.score:
            numFrames = 0
            
        if numFrames > 1000:
            game_over = True
        
        if game_over:
            print(f"Game Over! Your final score was: {game.score}")
            pygame.time.wait(2000)  # Wait for 2 seconds before resetting
            numFrames = 0
            # Reset game for next round
            key = pygame.K_RIGHT
            game.reset()
            game.update(key)
            game.draw()
            clock.tick(tickVal)
            prev_state = game.get_game_state()
            game.update(key)
            game.draw()
            clock.tick(tickVal)
            current_state = game.get_game_state()
            state = np.stack((prev_state, current_state), axis=-1)
        
        
        
        

if __name__ == '__main__':
    game = SnakeGame()
    run(game)