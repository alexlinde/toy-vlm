"""
Question generation module for the Toy VLM.
Generates questions, answers, and rationales for chain-of-thought reasoning.
"""

import random
from typing import Tuple, List, Dict, Any
from shapes import ShapeGenerator


class RationaleGenerator:
    """Generates structured rationales (program traces) for questions."""

    def __init__(self):
        self.shape_gen = ShapeGenerator()

    def count_shapes(self, metadata_list: List[Dict[str, Any]], target_shape: str) -> int:
        """Count how many shapes of a given type are in the image."""
        return sum(1 for m in metadata_list if m['shape'] == target_shape)

    def count_colors(self, metadata_list: List[Dict[str, Any]], target_color: str) -> int:
        """Count how many shapes of a given color are in the image."""
        return sum(1 for m in metadata_list if m['color'] == target_color)

    def count_sizes(self, metadata_list: List[Dict[str, Any]], target_size: str) -> int:
        """Count how many shapes of a given size are in the image."""
        return sum(1 for m in metadata_list if m['size_category'] == target_size)

    def exists(self, metadata_list: List[Dict[str, Any]], shape: str) -> bool:
        """Check if a shape exists in the image."""
        return any(m['shape'] == shape for m in metadata_list)

    def generate_identification_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Generate: 'what shapes are there?' type questions."""
        if not metadata_list:
            return None, None, None

        shapes = [m['shape'] for m in metadata_list]
        unique_shapes = list(set(shapes))

        question = "what shapes do you see"
        answer = " and ".join(unique_shapes)
        rationale = "look at image"

        return question, answer, rationale

    def generate_counting_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Generate: 'how many circles are there?' type questions."""
        if not metadata_list:
            return None, None, None

        target_shape = random.choice(self.shape_gen.get_available_shapes())
        count = self.count_shapes(metadata_list, target_shape)

        question = f"how many {target_shape}s are there"
        answer = str(count)

        # Rationale without the answer
        rationale = f"count {target_shape}s"

        return question, answer, rationale

    def generate_comparison_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Generate: 'are there more circles than squares?' type questions."""
        if not metadata_list:
            return None, None, None

        shapes = self.shape_gen.get_available_shapes()
        shape1, shape2 = random.sample(shapes, 2)

        count1 = self.count_shapes(metadata_list, shape1)
        count2 = self.count_shapes(metadata_list, shape2)

        question = f"are there more {shape1}s than {shape2}s"

        if count1 > count2:
            answer = "yes"
            comparison = "greater"
        elif count1 < count2:
            answer = "no"
            comparison = "less"
        else:
            answer = "no they are equal"
            comparison = "equal"

        # Rationale without answer
        rationale = f"count {shape1}s is {count1} count {shape2}s is {count2}"

        return question, answer, rationale

    def generate_existence_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Generate: 'is there a circle?' type questions."""
        if not metadata_list:
            return None, None, None

        # Randomly check for existing or non-existing shape
        if random.random() < 0.5 and metadata_list:
            # Ask about existing shape
            target = random.choice(metadata_list)
            shape = target['shape']
            exists = True
        else:
            # Ask about potentially non-existing shape
            shape = random.choice(self.shape_gen.get_available_shapes())
            exists = self.exists(metadata_list, shape)

        question = f"is there a {shape}"
        answer = "yes" if exists else "no"
        rationale = f"look for {shape}"

        return question, answer, rationale

    def generate_color_counting_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Generate: 'how many red shapes are there?' type questions."""
        if not metadata_list:
            return None, None, None

        target_color = random.choice(self.shape_gen.get_available_colors())
        count = self.count_colors(metadata_list, target_color)

        question = f"how many {target_color} shapes are there"
        answer = str(count)
        rationale = f"count {target_color} shapes"

        return question, answer, rationale

    def generate_size_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Generate: 'are there any large shapes?' type questions."""
        if not metadata_list:
            return None, None, None

        target_size = random.choice(self.shape_gen.get_available_sizes())
        count = self.count_sizes(metadata_list, target_size)
        exists = count > 0

        question = f"are there any {target_size} shapes"
        answer = "yes" if exists else "no"
        rationale = f"look for {target_size} shapes"

        return question, answer, rationale

    def generate_qa_with_rationale(self, metadata_list: List[Dict[str, Any]], difficulty: str = 'easy') -> Tuple[str, str, str]:
        """Generate question, answer, and rationale based on difficulty level.

        Args:
            metadata_list: List of shape metadata from image
            difficulty: 'easy', 'medium', or 'hard'

        Returns:
            (question, answer, rationale) tuple
        """
        if difficulty == 'easy':
            # Simple identification or existence
            generators = [
                self.generate_existence_qa,
                self.generate_identification_qa,
            ]
        elif difficulty == 'medium':
            # Counting and single-hop reasoning
            generators = [
                self.generate_counting_qa,
                self.generate_color_counting_qa,
                self.generate_size_qa,
            ]
        else:  # hard
            # Multi-hop reasoning
            generators = [
                self.generate_comparison_qa,
            ]

        # Keep trying until we get a valid result
        for _ in range(10):
            generator = random.choice(generators)
            result = generator(metadata_list)
            if result[0] is not None:
                return result

        # Fallback to existence question
        return self.generate_existence_qa(metadata_list)
    