"""
Question generation module for the Toy VLM.
Generates questions, answers, and rationales for chain-of-thought reasoning.
"""

import random
from typing import Tuple, List, Dict, Any
from shapes import ShapeGenerator, IMAGE_SIZE

class RationaleGenerator:
    """Generates structured rationales (program traces) for questions."""

    def __init__(self):
        self.shape_gen = ShapeGenerator()

    def count_shapes(self, metadata_list: List[Dict[str, Any]], target_shape: str) -> int:
        """Count how many shapes of a given type are in the image."""
        return sum(1 for m in metadata_list if m['shape'] == target_shape)

    def count_sizes(self, metadata_list: List[Dict[str, Any]], target_size: str) -> int:
        """Count how many shapes of a given size are in the image."""
        return sum(1 for m in metadata_list if m['size_category'] == target_size)

    def exists(self, metadata_list: List[Dict[str, Any]], shape: str) -> bool:
        """Check if a shape exists in the image."""
        return any(m['shape'] == shape for m in metadata_list)

    def generate_counting_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Generate: 'how many circles are there?' type questions."""
        if not metadata_list:
            return None, None, None

        target_shape = random.choice(self.shape_gen.get_available_shapes())
        count = self.count_shapes(metadata_list, target_shape)

        question = f"how many {target_shape} are there"
        answer = str(count)

        # Rationale without the answer
        rationale = f"count {target_shape} is {count}"

        return question, answer, rationale

    def generate_comparison_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Generate: 'are there more circles than squares?' type questions."""
        if not metadata_list:
            return None, None, None

        shapes = self.shape_gen.get_available_shapes()
        shape1, shape2 = random.sample(shapes, 2)

        count1 = self.count_shapes(metadata_list, shape1)
        count2 = self.count_shapes(metadata_list, shape2)

        question = f"are there more {shape1} than {shape2}"

        if count1 > count2:
            answer = "yes"
            comparison = "greater"
        elif count1 < count2:
            answer = "no"
            comparison = "less"
        else:
            answer = "no"
            comparison = "equal"

        # Rationale without answer
        rationale = f"count {shape1} is {count1} count {shape2} is {count2} which is {comparison}"

        return question, answer, rationale

    def generate_existence_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Generate: 'is there a circle?' type questions."""
        if not metadata_list:
            return None, None, None

        shape = random.choice(self.shape_gen.get_available_shapes())
        count = self.count_shapes(metadata_list, shape)

        question = f"is there a {shape}"
        answer = "yes" if count > 0 else "no"
        rationale = f"count {shape} is {count}"

        return question, answer, rationale

    def generate_size_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Generate: 'are there any large shapes?' type questions."""
        if not metadata_list:
            return None, None, None

        target_size = random.choice(self.shape_gen.get_available_sizes())
        count = self.count_sizes(metadata_list, target_size)

        question = f"are there any {target_size} shapes"
        answer = "yes" if count > 0 else "no"
        rationale = f"count {target_size} is {count}"

        return question, answer, rationale

    SIDES = ["left", "right", "top", "bottom"]
    def in_side(self, m: Dict[str, Any], side: str) -> bool:
        half = IMAGE_SIZE // 2
        cx = m.get('cx', m.get('center_x'))
        cy = m.get('cy', m.get('center_y'))
        if cx is None or cy is None:
            return False
        if side == "left":
            return cx < half
        if side == "right":
            return cx >= half
        if side == "top":
            return cy < half
        return cy >= half  # bottom

    def generate_positional_existence_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Generate: 'is there a circle on the left/right/top/bottom?' type questions."""
        if not metadata_list:
            return None, None, None

        target_shape = random.choice(self.shape_gen.get_available_shapes())
        side = random.choice(self.SIDES)
        count = sum(1 for m in metadata_list if m['shape'] == target_shape and self.in_side(m, side))

        question = random.choice([  
            f"is there a {target_shape} on the {side}", 
            f"are there any {target_shape} on the {side}"
        ])
        answer = "yes" if count > 0 else "no"
        rationale = f"count {target_shape} on {side} is {count}"

        return question, answer, rationale

    def generate_relative_position_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Generate: 'is a circle left of a square?' or 'above/below/right of' questions."""
        if not metadata_list:
            return None, None, None

        shapes = self.shape_gen.get_available_shapes()
        shape1, shape2 = random.sample(shapes, 2)
        relation = random.choice(["left of", "right of", "above", "below"])

        objs1 = [m for m in metadata_list if m['shape'] == shape1]
        objs2 = [m for m in metadata_list if m['shape'] == shape2]

        def satisfies(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            ax = a.get('cx', a.get('center_x'))
            ay = a.get('cy', a.get('center_y'))
            bx = b.get('cx', b.get('center_x'))
            by = b.get('cy', b.get('center_y'))
            if None in (ax, ay, bx, by):
                return False
            if relation == "left of":
                return ax < bx
            if relation == "right of":
                return ax > bx
            if relation == "above":
                return ay < by
            return ay > by  # below

        exists_pair = any(satisfies(a, b) for a in objs1 for b in objs2)

        question = f"is a {shape1} {relation} a {shape2}"
        answer = "yes" if exists_pair else "no"
        rationale = f"check pairs of {shape1} and {shape2} for relation {relation}"

        return question, answer, rationale

    def generate_side_count_comparison_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Generate: 'are there more circles on the left than the right?' type questions."""
        if not metadata_list:
            return None, None, None

        target_shape = random.choice(self.shape_gen.get_available_shapes())
        half = IMAGE_SIZE // 2

        left_count = sum(1 for m in metadata_list if m['shape'] == target_shape and (m.get('cx', m.get('center_x')) or 0) < half)
        right_count = sum(1 for m in metadata_list if m['shape'] == target_shape and (m.get('cx', m.get('center_x')) or 0) >= half)

        side = random.choice(["left", "right"])
        question = f"are there more {target_shape} on the {side}"
        if side == "left" and left_count > right_count:
            answer = "yes"
            comparison = "greater"
            not_side = "right"
        elif side == "left" and left_count < right_count:
            answer = "no"
            comparison = "less"
            not_side = "right"
        elif side == "right" and left_count < right_count:
            answer = "yes"
            comparison = "greater"
            not_side = "left"
        elif side == "right" and left_count > right_count:
            answer = "no"
            comparison = "less"
            not_side = "left"
        elif side == "left":
            answer = "no"
            comparison = "equal"
            not_side = "right"
        else:
            answer = "no"
            comparison = "equal"
            not_side = "left"

        ## todo: fix this, it's wrong
        rationale = f"count {target_shape} on {side} is {left_count} count {target_shape} on {not_side} is {right_count} which is {comparison}"

        return question, answer, rationale

    def generate_compositional_positional_qa(self, metadata_list: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Generate compositional: 'is a circle left of the square that is above the triangle?'"""
        if not metadata_list:
            return None, None, None

        shapes = self.shape_gen.get_available_shapes()
        # If fewer than 3 shape types exist in the scene, fall back to relative position
        shape1, shape2, shape3 = random.sample(shapes, 3)
        relation1 = random.choice(["left of", "right of", "above", "below"])
        relation2 = random.choice(["left of", "right of", "above", "below"])

        objs1 = [m for m in metadata_list if m['shape'] == shape1]
        objs2 = [m for m in metadata_list if m['shape'] == shape2]
        objs3 = [m for m in metadata_list if m['shape'] == shape3]

        def rel(a: Dict[str, Any], b: Dict[str, Any], r: str) -> bool:
            ax = a.get('cx', a.get('center_x'))
            ay = a.get('cy', a.get('center_y'))
            bx = b.get('cx', b.get('center_x'))
            by = b.get('cy', b.get('center_y'))
            if None in (ax, ay, bx, by):
                return False
            if r == "left of":
                return ax < bx
            if r == "right of":
                return ax > bx
            if r == "above":
                return ay < by
            return ay > by  # below

        exists_triple = any(rel(a, b, relation1) and rel(b, c, relation2) for a in objs1 for b in objs2 for c in objs3)

        question = (
            f"is a {shape1} {relation1} the {shape2} that is {relation2} the {shape3}"
        )
        answer = "yes" if exists_triple else "no"
        rationale = (
            f"find {shape2} and {shape3} with relation {relation2} then check {shape1} {relation1} that {shape2}"
        )

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
            # Direct perception (existence and positional)
            generators = [
                self.generate_existence_qa,
                self.generate_positional_existence_qa,
            ]
        elif difficulty == 'medium':
            # Multi-step comparisons (counts/relative positions)
            generators = [
                self.generate_counting_qa,
                self.generate_size_qa,
                self.generate_relative_position_qa,
                self.generate_side_count_comparison_qa,
            ]
        else:  # hard
            # Compositional multi-step reasoning
            generators = [
                self.generate_comparison_qa,
                self.generate_compositional_positional_qa,
            ]

        # Keep trying until we get a valid result
        for _ in range(10):
            generator = random.choice(generators)
            result = generator(metadata_list)
            if result[0] is not None:
                return result

        # Fallback to existence question
        return self.generate_existence_qa(metadata_list)
    