"""
Shape generation module for the Toy VLM.
Handles creation of geometric shapes and related functionality.
"""

import numpy as np
import random
from typing import List, Tuple
from PIL import Image

# Image constants
IMAGE_SIZE = 64

class ShapeGenerator:
    """Generates simple geometric shapes as images."""
    
    def __init__(self):
        self.shapes = ['square', 'circle', 'rectangle', 'cross', 'triangle']
    
    def generate_shape_image(self, shape_type: str, add_noise: bool) -> np.ndarray:
        """Generate an image with a single shape."""
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        
        # Random position and size
        margin = 10
        max_size = IMAGE_SIZE - 2 * margin
        size = random.randint(15, max_size)
        
        if shape_type == 'square':
            x = random.randint(margin, IMAGE_SIZE - margin - size)
            y = random.randint(margin, IMAGE_SIZE - margin - size)
            img[y:y+size, x:x+size] = 1.0
            
        elif shape_type == 'circle':
            center_x = random.randint(margin + size//2, IMAGE_SIZE - margin - size//2)
            center_y = random.randint(margin + size//2, IMAGE_SIZE - margin - size//2)
            radius = size // 2
            
            y, x = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            img[mask] = 1.0
            
        elif shape_type == 'rectangle':
            min_side = 15
            # Ensure rectangle is meaningfully non-square
            while True:
                width = random.randint(min_side, max_size)
                height = random.randint(min_side, max_size)
                ratio = max(width, height) / max(1, min(width, height))
                if ratio >= 1.3:
                    break
            x = random.randint(margin, IMAGE_SIZE - margin - width)
            y = random.randint(margin, IMAGE_SIZE - margin - height)
            img[y:y+height, x:x+width] = 1.0
            
        elif shape_type == 'cross':
            cx = IMAGE_SIZE // 2 + random.randint(-10, 10)
            cy = IMAGE_SIZE // 2 + random.randint(-10, 10)
            thickness = random.randint(3, 7)
            length = random.randint(15, 30)
            
            # Horizontal line
            img[cy-thickness:cy+thickness, cx-length:cx+length] = 1.0
            # Vertical line
            img[cy-length:cy+length, cx-thickness:cx+thickness] = 1.0
            
        elif shape_type == 'triangle':
            # Generate proper triangle with three vertices
            size = random.randint(24, max_size)
            cx = random.randint(margin + size//2, IMAGE_SIZE - margin - size//2)
            cy = random.randint(margin + size//2, IMAGE_SIZE - margin - size//2)
            
            # Define three vertices of the triangle
            # Top vertex
            x1, y1 = cx, cy - size//2
            # Bottom left vertex
            x2, y2 = cx - size//2, cy + size//2
            # Bottom right vertex
            x3, y3 = cx + size//2, cy + size//2
            
            # Fill triangle using barycentric coordinates
            for y in range(max(0, cy - size//2), min(IMAGE_SIZE, cy + size//2 + 1)):
                for x in range(max(0, cx - size//2), min(IMAGE_SIZE, cx + size//2 + 1)):
                    # Calculate barycentric coordinates
                    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
                    if abs(denom) > 1e-10:  # Avoid division by zero
                        a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
                        b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
                        c = 1 - a - b
                        
                        # Point is inside triangle if all barycentric coordinates are non-negative
                        if a >= 0 and b >= 0 and c >= 0:
                            img[y, x] = 1.0
        
        # Apply random rotation
        rotation_angle = random.uniform(0, 360)
        pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
        rotated_pil = pil_img.rotate(rotation_angle, fillcolor=0)
        img = np.array(rotated_pil, dtype=np.float32) / 255.0
        
        # Add slight noise
        if add_noise:
            noise = np.random.normal(0, 0.05, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        return img
    
    def generate_random_shape(self, add_noise: bool = True) -> Tuple[str, np.ndarray]:
        """Generate a random shape and return its type and image."""
        shape_type = random.choice(self.shapes)
        image = self.generate_shape_image(shape_type, add_noise)
        return shape_type, image
    
    def get_available_shapes(self) -> List[str]:
        """Return list of available shape types."""
        return self.shapes.copy()

