"""
Evaluation suite for the Toy VLM with Chain-of-Thought reasoning.
Tests the model on various question types and measures exact match accuracy.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
from shapes import ShapeGenerator
from questions import RationaleGenerator
from text import TextProcessor, SimpleTokenizer
from model import ToyVLM, generate_response, DEVICE
import random

class VLMEvaluator:
    """Evaluates VLM on generated test sets."""

    def __init__(self, model, text_processor):
        self.model = model
        self.text_processor = text_processor
        self.shape_gen = ShapeGenerator()
        self.rationale_gen = RationaleGenerator()

    def generate_test_set(self, num_samples: int, difficulty: str) -> List[Dict]:
        """Generate a test set with ground truth."""
        test_samples = []

        for _ in range(num_samples):
            # Generate multi-shape image
            num_shapes = random.randint(1, 4)
            image, metadata = self.shape_gen.generate_multi_shape_image(num_shapes, False)

            # Generate question with ground truth rationale and answer
            question, answer, rationale = self.rationale_gen.generate_qa_with_rationale(
                metadata, difficulty=difficulty
            )

            test_samples.append({
                'image': image,
                'metadata': metadata,
                'question': question,
                'ground_truth_answer': answer,
                'ground_truth_rationale': rationale
            })

        return test_samples

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        return answer.lower().strip().replace('.', '').replace(',', '')

    def evaluate_exact_match(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth (exact match)."""
        return self.normalize_answer(predicted) == self.normalize_answer(ground_truth)

    def evaluate_test_set(self, test_samples: List[Dict], show_examples: int) -> Tuple[Dict[str, float], List[Dict]]:
        """Evaluate model on test set and return metrics."""
        self.model.eval()

        correct = 0
        total = len(test_samples)
        empty_predictions = 0
        generation_errors = 0

        results = []

        print(f"\nEvaluating on {total} samples...")
        for i, sample in enumerate(tqdm(test_samples)):
            image = sample['image']
            question = sample['question']
            gt_answer = sample['ground_truth_answer']

            # Grayscale image: (H,W) -> (1,H,W)
            img = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0 # (1,H,W)

            # Generate prediction
            try:
                pred_rationale, pred_answer = generate_response(
                    self.model, img, question, max_length=35, return_rationale=True
                )
                if not pred_answer.strip():
                    empty_predictions += 1
            except Exception as e:
                print(f"\nError generating response for question: {question}")
                print(f"Error: {e}")
                pred_rationale, pred_answer = "", ""
                generation_errors += 1

            # Check exact match
            is_correct = self.evaluate_exact_match(pred_answer, gt_answer)
            if is_correct:
                correct += 1

            results.append({
                'question': question,
                'ground_truth_answer': gt_answer,
                'predicted_answer': pred_answer,
                'ground_truth_rationale': sample['ground_truth_rationale'],
                'predicted_rationale': pred_rationale,
                'correct': is_correct
            })

            # Show examples
            if i < show_examples:
                print(f"\n--- Example {i+1} ---")
                print(f"Question: {question}")
                print(f"GT Answer: {gt_answer}")
                print(f"Pred Answer: {pred_answer}")
                print(f"GT Rationale: {sample['ground_truth_rationale']}")
                print(f"Pred Rationale: {pred_rationale}")
                print(f"Correct: {is_correct}")

        accuracy = correct / total if total > 0 else 0.0

        metrics = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'empty_predictions': empty_predictions,
            'generation_errors': generation_errors,
            'empty_rate': empty_predictions / total if total > 0 else 0.0,
            'error_rate': generation_errors / total if total > 0 else 0.0
        }

        return metrics, results

    def evaluate_by_difficulty(self, num_samples_per_difficulty: int):
        """Evaluate on all difficulty levels."""
        difficulties = ['easy', 'medium', 'hard']
        all_metrics = {}

        for difficulty in difficulties:
            print(f"\n{'='*60}")
            print(f"Evaluating on {difficulty.upper()} questions")
            print('='*60)

            test_set = self.generate_test_set(num_samples_per_difficulty, difficulty)
            metrics, results = self.evaluate_test_set(test_set, show_examples=5)

            all_metrics[difficulty] = metrics

            print(f"\n{difficulty.upper()} Results:")
            print(f"  Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
            if metrics.get('empty_predictions', 0) > 0:
                print(f"  Empty predictions: {metrics['empty_predictions']} ({metrics['empty_rate']:.1%})")
            if metrics.get('generation_errors', 0) > 0:
                print(f"  Generation errors: {metrics['generation_errors']} ({metrics['error_rate']:.1%})")

        # Overall summary
        print(f"\n{'='*60}")
        print("OVERALL SUMMARY")
        print('='*60)
        total_correct = sum(all_metrics[d]['correct'] for d in difficulties)
        total_samples = sum(all_metrics[d]['total'] for d in difficulties)
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        for difficulty in difficulties:
            metrics = all_metrics[difficulty]
            print(f"{difficulty.capitalize():8s}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")

        print(f"{'Overall':8s}: {overall_accuracy:.2%} ({total_correct}/{total_samples})")

        return all_metrics


def main():
    """Run evaluation on trained model."""
    print("Loading model for evaluation...")

    # Load tokenizer
    tokenizer = SimpleTokenizer(vocab_file='tokenizer_vocab.json')
    text_processor = TextProcessor()
    text_processor.tokenizer = tokenizer

    # Create model
    model = ToyVLM(text_processor, num_layers=6)

    # Load trained weights
    try:
        model.load_state_dict(torch.load('toy_vlm_cot.pth', map_location=DEVICE))
        print("Loaded model weights from 'toy_vlm_cot.pth'")
    except FileNotFoundError:
        print("Warning: Model weights not found. Using untrained model.")

    model = model.to(DEVICE)
    model.eval()

    # Create evaluator
    evaluator = VLMEvaluator(model, text_processor)

    # Run evaluation
    all_metrics = evaluator.evaluate_by_difficulty(num_samples_per_difficulty=10)

    return all_metrics


if __name__ == "__main__":
    main()