"""
Shape generation module for the Toy VLM.
Handles creation of geometric shapes and related functionality.
"""

import numpy as np
import random
from typing import List, Tuple

# Image constants
IMAGE_SIZE = 64

class ShapeGenerator:
    """Generates simple geometric shapes as images."""
    
    def __init__(self):
        self.shapes = ['square', 'circle', 'rectangle', 'cross', 'triangle']
    
    def generate_shape_image(self, shape_type: str) -> np.ndarray:
        """Generate a 64x64 image with a single shape."""
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
            width = random.randint(15, max_size)
            height = random.randint(15, max_size)
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
            # Simple triangle using three points
            size = random.randint(20, max_size)
            cx = random.randint(margin + size//2, IMAGE_SIZE - margin - size//2)
            cy = random.randint(margin + size//2, IMAGE_SIZE - margin - size//2)
            
            # Create triangle mask
            for y in range(max(0, cy - size//2), min(IMAGE_SIZE, cy + size//2)):
                for x in range(max(0, cx - size//2), min(IMAGE_SIZE, cx + size//2)):
                    # Simple triangle condition
                    if (y - cy + size//2) > 0 and abs(x - cx) < (cy + size//2 - y):
                        img[y, x] = 1.0
        
        # Add slight noise
        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)
        
        return img
    
    def generate_random_shape(self) -> Tuple[str, np.ndarray]:
        """Generate a random shape and return its type and image."""
        shape_type = random.choice(self.shapes)
        image = self.generate_shape_image(shape_type)
        return shape_type, image
    
    def get_available_shapes(self) -> List[str]:
        """Return list of available shape types."""
        return self.shapes.copy()

