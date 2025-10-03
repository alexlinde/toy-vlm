"""
Shape generation module for the Toy VLM.
Handles creation of geometric shapes and related functionality.
"""

import numpy as np
import random
from enum import Enum
from typing import List, Tuple, Dict, Any
from PIL import Image, ImageDraw

# Image constants
IMAGE_SIZE = 64

class ObjType(Enum):
    """Object type enumeration for shape classification."""
    SQUARE = "square"
    CIRCLE = "circle"
    # RECTANGLE = "rectangle"
    CROSS = "cross"
    TRIANGLE = "triangle"

class ObjSize(Enum):
    """Object size enumeration for shape classification."""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

SIZE_RANGES = {
    ObjSize.SMALL: (8, 15),
    ObjSize.MEDIUM: (16, 25),
    ObjSize.LARGE: (26, 35)
}

class ShapeGenerator:
    """Generates simple geometric shapes as grayscale images."""

    def _draw_single_shape(self, img: np.ndarray, shape_type: ObjType, size: int, cx: int, cy: int, rotation: float = 0.0) -> Dict[str, Any]:
        """Draw a single shape on the grayscale image and return its metadata."""
        # Create a temporary single-channel mask image for the shape
        shape_mask = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), 0)
        draw = ImageDraw.Draw(shape_mask)

        if shape_type == ObjType.SQUARE:
            half = size // 2
            x1, y1 = cx - half, cy - half
            x2, y2 = cx + half, cy + half
            draw.rectangle([x1, y1, x2, y2], fill=255)

        elif shape_type == ObjType.CIRCLE:
            radius = size // 2
            x1, y1 = cx - radius, cy - radius
            x2, y2 = cx + radius, cy + radius
            draw.ellipse([x1, y1, x2, y2], fill=255)

        elif hasattr(ObjType, 'RECTANGLE') and shape_type == ObjType.RECTANGLE:
            width = int(size * random.uniform(0.6, 1.4))
            height = int(size * random.uniform(0.6, 1.4))
            x1, y1 = cx - width // 2, cy - height // 2
            x2, y2 = cx + width // 2, cy + height // 2
            draw.rectangle([x1, y1, x2, y2], fill=255)

        elif shape_type == ObjType.CROSS:
            thickness = max(2, size // 8)
            length = size // 2
            # Horizontal bar
            hx1, hy1 = cx - length, cy - thickness
            hx2, hy2 = cx + length, cy + thickness
            draw.rectangle([hx1, hy1, hx2, hy2], fill=255)
            # Vertical bar
            vx1, vy1 = cx - thickness, cy - length
            vx2, vy2 = cx + thickness, cy + length
            draw.rectangle([vx1, vy1, vx2, vy2], fill=255)

        elif shape_type == ObjType.TRIANGLE:
            half = size // 2
            x1, y1 = cx, cy - half  # top
            x2, y2 = cx - half, cy + half  # bottom-left
            x3, y3 = cx + half, cy + half  # bottom-right
            draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=255)

        # Rotate the mask around the shape center and composite onto the image
        if rotation:
            shape_mask = shape_mask.rotate(rotation, resample=Image.NEAREST, expand=False, center=(cx, cy))

        mask_np = np.array(shape_mask, dtype=np.uint8)
        img[mask_np > 0] = 255

        metadata = {
            'shape': shape_type.value,
            'size': size,
            'cx': cx,
            'cy': cy,
            'rotation': float(rotation)
        }

        return metadata

    def generate_multi_shape_image(self, num_shapes: int, add_noise: bool) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Generate a 64x64 grayscale image with multiple shapes and return metadata.

        Returns:
            image: Grayscale numpy array of shape (64, 64) with values 0-255
            metadata_list: List of dicts containing shape info (shape, size, center_x, center_y)
        """

        # Initialize grayscale image (black background)
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        metadata_list = []

        # Track occupied regions to avoid too much overlap
        occupied = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)

        attempts = 0
        max_attempts = num_shapes * 10

        while len(metadata_list) < num_shapes and attempts < max_attempts:
            attempts += 1

            # Random shape and size
            shape_type = random.choice(list(ObjType))
            size_category = random.choice(list(ObjSize))
            size_min, size_max = SIZE_RANGES[size_category]
            size = random.randint(size_min, size_max)

            # Random position with margin
            margin = size // 2 + 5
            if margin >= IMAGE_SIZE // 2:
                continue

            cx = random.randint(margin, IMAGE_SIZE - margin)
            cy = random.randint(margin, IMAGE_SIZE - margin)

            # Check overlap (no overlap allowed for clearer images)
            check_radius = size // 2 + 3  # Add small margin
            y1, y2 = max(0, cy - check_radius), min(IMAGE_SIZE, cy + check_radius)
            x1, x2 = max(0, cx - check_radius), min(IMAGE_SIZE, cx + check_radius)
            overlap_ratio = occupied[y1:y2, x1:x2].sum() / max(1, (y2-y1) * (x2-x1))

            if overlap_ratio > 0.0:  # No overlap allowed
                continue

            # Draw the shape (grayscale - no color parameter)
            metadata = self._draw_single_shape(img, shape_type, size, cx, cy)
            metadata['size_category'] = size_category.value
            metadata_list.append(metadata)

            # Mark region as occupied
            occupied[y1:y2, x1:x2] = True

        # Add slight noise (for grayscale, scale is 0-255)
        if add_noise:
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img, metadata_list


    def get_available_shapes(self) -> List[str]:
        """Return list of available shape types."""
        return [shape.value for shape in ObjType]


    def get_available_sizes(self) -> List[str]:
        """Return list of available size categories."""
        return [size.value for size in ObjSize]

