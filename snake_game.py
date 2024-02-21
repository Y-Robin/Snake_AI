import pygame
import random
import numpy as np
import cv2

# Scaling factor to adjust the game's resolution and element sizes
gamesizeMultiplier = 0.5

class SnakeGame:
    def __init__(self):
        self.WIDTH, self.HEIGHT = int(600 * gamesizeMultiplier), int(600 * gamesizeMultiplier)
        self.WIDTHM, self.HEIGHTM = 200, 200
        self.BLOCK_SIZE = int(30 * gamesizeMultiplier)
        self.WHITE = (255, 0, 0)
        self.RED = (170, 0, 0)
        self.GREEN = (120, 0, 0)
        self.BLACK = (0, 0, 0)
        
        pygame.init()
        self.font = pygame.font.SysFont('Arial', 25)
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('Snake Game')
        
        self.clock = pygame.time.Clock()
        self.snake = [(self.WIDTH / 2, self.HEIGHT / 2)]
        
        self.food = self.spawn_food()
        self.score = 0
        
        self.snake_direction = pygame.K_RIGHT
        
        self.opposite_directions = {
            pygame.K_UP: pygame.K_DOWN,
            pygame.K_DOWN: pygame.K_UP,
            pygame.K_LEFT: pygame.K_RIGHT,
            pygame.K_RIGHT: pygame.K_LEFT
        }
        self.oldDist = self.distanceHeadFood()
        self.newDist = self.distanceHeadFood()
        
        self.gameOver = False
        self.gameOverSelf = False
        
        self.ateFood = False
        
        # Initial drawing of game elements
        self.draw()
        
        
        
    def distanceHeadFood(self):
        """Calculate the Manhattan distance between two points."""
        head_x, head_y = self.snake[0]
        return abs(head_x - self.food[0]) + abs(head_y - self.food[1])

    def spawn_food(self):
        """Places food at a random location on the screen, not overlapping the snake."""
        while True:
            food_position = (random.randrange(2, self.WIDTH // self.BLOCK_SIZE-1) * self.BLOCK_SIZE,
                             random.randrange(2, self.HEIGHT // self.BLOCK_SIZE-1) * self.BLOCK_SIZE)
            if food_position not in self.snake:
                return food_position
                

    def update(self, key):
        """Updates the snake's position and checks for game over conditions."""
        self.ateFood = False
        gameOverL = False
        head_x, head_y = self.snake[0]
        if key == self.opposite_directions.get(self.snake_direction):
            if self.snake_direction == pygame.K_UP:
                head_y -= self.BLOCK_SIZE
            elif self.snake_direction == pygame.K_DOWN:
                head_y += self.BLOCK_SIZE
            elif self.snake_direction == pygame.K_LEFT:
                head_x -= self.BLOCK_SIZE
            elif self.snake_direction == pygame.K_RIGHT:
                head_x += self.BLOCK_SIZE
        else:
            self.snake_direction = key
            if key == pygame.K_UP:
                head_y -= self.BLOCK_SIZE
            elif key == pygame.K_DOWN:
                head_y += self.BLOCK_SIZE
            elif key == pygame.K_LEFT:
                head_x -= self.BLOCK_SIZE
            elif key == pygame.K_RIGHT:
                head_x += self.BLOCK_SIZE
            
         # Game over on boundary collision
        if head_x < self.BLOCK_SIZE or head_x >= self.WIDTH-self.BLOCK_SIZE or head_y < self.BLOCK_SIZE or head_y >= self.HEIGHT-self.BLOCK_SIZE or (head_x, head_y) in self.snake:
            self.gameOver = True
            gameOverL = True    
            
        if (head_x, head_y) in self.snake[:-1]:
            self.gameOver = True
            self.gameOverSelf = True
            gameOverL = True  

        new_head = (head_x, head_y)
        self.snake.insert(0, new_head)
        
        # Check if snake eats food
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
            self.ateFood = True
        else:
            self.snake.pop()
        
        self.oldDist = self.newDist
        self.newDist = self.distanceHeadFood()
        

        
        return gameOverL

    def draw(self):
        """Draws the game state to the screen."""
        self.screen.fill(self.BLACK)
        idx = 1
        for block in self.snake:
            if idx == 1:
                pygame.draw.rect(self.screen, self.RED, pygame.Rect(block[0], block[1], self.BLOCK_SIZE, self.BLOCK_SIZE))
                idx += 1
            else:
                pygame.draw.rect(self.screen, self.GREEN, pygame.Rect(block[0], block[1], self.BLOCK_SIZE, self.BLOCK_SIZE))
        pygame.draw.rect(self.screen, self.WHITE, pygame.Rect(self.food[0], self.food[1], self.BLOCK_SIZE, self.BLOCK_SIZE))
        
        # Capture the current frame as image data for processing
        self.image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        self.image_data = np.transpose(self.image_data, (1, 0, 2))  # Transpose the image data
        
        score_text = self.font.render(f'Score: {self.score}', True, self.WHITE)
        self.screen.blit(score_text, [0, 0])
        pygame.display.flip()
        
    def get_game_state(self):
        """Capture and return the current game state as an image."""
        # Resize the captured image data for model processing
        retIm = cv2.resize(self.image_data[:,:,0], (self.WIDTHM, self.HEIGHTM ), interpolation=cv2.INTER_CUBIC)
        return retIm
        
    def calculate_rewards(self):
        head_x, head_y = self.snake[0]
        if self.gameOverSelf:
            return -100
        elif self.gameOver:
            return -50
        elif self.ateFood:
            return 50
        elif self.oldDist > self.newDist:
            return 10
        elif self.oldDist < self.newDist:
            return -5
        
        return 0
    
    def reset(self):
        """Resets the game to the initial state."""
        self.snake = [(self.WIDTH / 2, self.HEIGHT / 2)]
        self.snake_direction = pygame.K_RIGHT
        self.food = self.spawn_food()
        self.score = 0
        self.gameOver = False
        self.gameOverSelf = False
