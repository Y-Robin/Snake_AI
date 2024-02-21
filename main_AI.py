from snake_game import SnakeGame
import pygame
import numpy as np
#import time
from model import DQNModel  # Make sure this is the path to your model file
import matplotlib.pyplot as plt


numColArray = []
numColBest = 0
tickVal = 2000
def run(game):
    """Main function to run the Pong game with DQN model."""
    # For plotting
    plt.ion()
    fig, ax = plt.subplots()
    plt.show()
    # Initialize game
    game = SnakeGame()
    clock = pygame.time.Clock()
    stateBib = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]
    
    # Model setup
    input_shape = (game.HEIGHTM, game.WIDTHM, 2)  # Input shape for the model
    action_size = 4  # Number of possible actions (up, down)
    model = DQNModel(input_shape, action_size,True) # Initialize the model
    numColBest = model.MaxScore
    batch_size = 32
    
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
    
    running = True
    numFrames = 0
    key = pygame.K_RIGHT
    while running:
        numFrames +=1
        oldGameScore = game.score
        # Quit if window is closed
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        
        # Model prediction 
        action = model.predict(state)
        
        # Update game state based on model predictions
        game_over = game.update(stateBib[action])
        
        # Draw/update the game screen
        game.draw()
        clock.tick(tickVal)
        
        # Prepare for the next state
        prev_state = current_state
        current_state = game.get_game_state()
        stateOld = state
        state = np.stack((prev_state, current_state), axis=-1)
        
        # Calculate rewards and remember experiences
        reward = game.calculate_rewards()
        if oldGameScore < game.score:
            numFrames = 0
            
        if numFrames > 1000:
            reward = -10
            game_over = True
        model.remember(stateOld, action, reward, state, game_over)  # For left paddle
        

        
        # Handle game over and training
        if game_over:
            print("----")
            # Log score
            numColArray.append(game.score) 
            
            # Save model if score improved
            if game.score > numColBest:
                model.save_model_and_parameters(game.score)
                numColBest = game.score
            
            #Plot Scores
            ax.clear()  # Clear the current plot
            ax.plot(numColArray)  # Plot the updated array
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)
            
            
            print(f"Game Over. Score {game.score}")

            numFrames = 0
            
            # Train model and update target model if enough memory collected
            model.train(batch_size)
            model.update_target_model()
            
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

    pygame.quit()

if __name__ == '__main__':
    game = SnakeGame()
    run(game)