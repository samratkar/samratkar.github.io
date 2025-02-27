import pygame
import random
import time
import os

# Initialize Pygame
pygame.init()

# Set up the game window
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
GRID_SIZE = 40  # Increased size to make images more visible
GRID_WIDTH = WINDOW_WIDTH // GRID_SIZE
GRID_HEIGHT = WINDOW_HEIGHT // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Load images
def load_and_scale_image(image_name):
    image_path = os.path.join('assets', image_name)
    try:
        image = pygame.image.load(image_path)
        return pygame.transform.scale(image, (GRID_SIZE, GRID_SIZE))
    except:
        print(f"Couldn't load {image_name}. Creating a default surface.")
        surface = pygame.Surface((GRID_SIZE, GRID_SIZE))
        if image_name == 'snake.png':
            surface.fill((0, 255, 0))  # Green for snake
        else:
            surface.fill((255, 0, 0))  # Red for apple
        return surface

# Create the game window
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption('Snake Game')

class Snake:
    def __init__(self):
        self.positions = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = (1, 0)
        self.grow = False
        self.image = load_and_scale_image('snake.png')
        # Create rotated versions of the snake image
        self.images = {
            (1, 0): self.image,  # Right
            (-1, 0): pygame.transform.rotate(self.image, 180),  # Left
            (0, -1): pygame.transform.rotate(self.image, 90),  # Up
            (0, 1): pygame.transform.rotate(self.image, 270)  # Down
        }

    def move(self):
        current = self.positions[0]
        x, y = self.direction
        new = (((current[0] + x) % GRID_WIDTH), (current[1] + y) % GRID_HEIGHT)
        
        if new in self.positions[:-1]:
            return False
        
        self.positions.insert(0, new)
        if not self.grow:
            self.positions.pop()
        else:
            self.grow = False
        return True

    def change_direction(self, direction):
        x, y = direction
        current_x, current_y = self.direction
        if x * current_x + y * current_y != 0:
            return
        self.direction = direction

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.image = load_and_scale_image('apple.png')
        self.spawn_food()

    def spawn_food(self):
        x = random.randint(0, GRID_WIDTH - 1)
        y = random.randint(0, GRID_HEIGHT - 1)
        self.position = (x, y)

def main():
    clock = pygame.time.Clock()
    snake = Snake()
    food = Food()
    score = 0
    game_over = False

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    snake.change_direction((0, -1))
                elif event.key == pygame.K_DOWN:
                    snake.change_direction((0, 1))
                elif event.key == pygame.K_LEFT:
                    snake.change_direction((-1, 0))
                elif event.key == pygame.K_RIGHT:
                    snake.change_direction((1, 0))

        if not game_over:
            if not snake.move():
                game_over = True

            if snake.positions[0] == food.position:
                snake.grow = True
                score += 1
                food.spawn_food()

            # Draw everything
            window.fill(BLACK)
            
            # Draw food
            window.blit(food.image, 
                       (food.position[0] * GRID_SIZE,
                        food.position[1] * GRID_SIZE))

            # Draw snake
            for i, position in enumerate(snake.positions):
                if i == 0:  # Head
                    window.blit(snake.images[snake.direction],
                              (position[0] * GRID_SIZE,
                               position[1] * GRID_SIZE))
                else:  # Body
                    window.blit(snake.image,
                              (position[0] * GRID_SIZE,
                               position[1] * GRID_SIZE))

            pygame.display.flip()
            clock.tick(10)

    # Display game over screen
    font = pygame.font.Font(None, 74)
    text = font.render(f'Game Over! Score: {score}', True, WHITE)
    text_rect = text.get_rect(center=(WINDOW_WIDTH/2, WINDOW_HEIGHT/2))
    window.blit(text, text_rect)
    pygame.display.flip()
    time.sleep(2)
    
    pygame.quit()

if __name__ == '__main__':
    main()
