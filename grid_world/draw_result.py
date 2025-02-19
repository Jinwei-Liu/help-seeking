import pygame
import numpy as np

# Color function, returns color based on value
def value_to_color(value):
    # Start and end colors
    start_color = (62, 144, 191)  # RGB for start color
    end_color = (216, 211, 231)  # RGB for end color
    
    # Linear interpolation to calculate color value
    r = int(start_color[0] + (end_color[0] - start_color[0]) * value)
    g = int(start_color[1] + (end_color[1] - start_color[1]) * value)
    b = int(start_color[2] + (end_color[2] - start_color[2]) * value)
    
    return (r, g, b)  # Return RGB value

def draw_result(data, position, save_address):
    # Normalize data
    min_val = np.min(data)
    max_val = np.max(data)
    data_raw = data
    data = (data - min_val) / (max_val - min_val + 1e-6)

    # Initialize Pygame
    pygame.init()

    # Set up window
    window_size = 900
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("9x9 Grid")

    screen.fill((255, 255, 255))  # Clear screen
    values_raw = np.zeros((9, 9))
    values = np.zeros((9, 9))
    for i in range(9 * 9):
        values[i // 9, i % 9] = data[i][position[i]]
        values_raw[i // 9, i % 9] = data_raw[i][position[i]]

    # Draw grid
    grid_size = window_size // 9
    border_width = 1  # Border width
    border_radius = 10  # Border radius

    for row in range(9):
        for col in range(9):
            value = values[row, col]
            value_raw = values_raw[row, col]
            color = value_to_color(value)
            
            # Draw rounded rectangle with background color
            pygame.draw.rect(
                screen, color, 
                (col * grid_size, row * grid_size, grid_size, grid_size), 
                border_radius=border_radius
            )
            
            # Draw black rounded border
            pygame.draw.rect(
                screen, (100, 100, 100), 
                (col * grid_size, row * grid_size, grid_size, grid_size), 
                width=border_width, border_radius=border_radius
            )
            
            # Display value
            font = pygame.font.Font(None, 36)
            text = font.render(str(round(value_raw, 2)), True, (255, 255, 255))
            text_rect = text.get_rect(center=(col * grid_size + grid_size // 2, row * grid_size + grid_size // 2))
            screen.blit(text, text_rect)

    pygame.display.flip()

    # Save image
    pygame.image.save(screen, save_address)

