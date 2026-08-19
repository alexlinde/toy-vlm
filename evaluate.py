"""
Evaluation script for the Toy VLM.

Measures exact-match accuracy of greedy generation across every question
template in questions.txt, split into three families ('identification',
'yes', 'no'), alongside each template's and family's template-memorization
baseline (the accuracy achievable by memorizing the majority gold answer per
template, since gold is a pure function of the template). There is no
other quantitative evaluation in this project - test_model.py is a manual,
interactive GUI.
"""

import re
import random
import argparse
from collections import Counter

import numpy as np
import torch

from shapes import ShapeGenerator
from questions import QuestionGenerator
from model import load_trained_model, generate_response
from device import DEVICE

FAMILIES = ('identification', 'yes', 'no')


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for exact-match comparison."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())


def classify_family(question_generator, template_name: str) -> str:
    """Classify a template by probing which variable (if any) drives its question:
    'no' if it asks about random_other_shape, 'yes' if it asks about shape,
    otherwise it's an open identification question (e.g. 'what shape is this')."""
    template = question_generator.env.get_template(template_name)
    probe = template.render(shape='__SHAPE__', random_other_shape='__OTHER__').split('|', 1)[0]
    if '__OTHER__' in probe:
        return 'no'
    if '__SHAPE__' in probe:
        return 'yes'
    return 'identification'


def render_for_template(question_generator, template_name: str, shape_type: str):
    """Render one specific template for a given shape. QuestionGenerator.generate_qa_pair
    only supports a randomly-chosen template, so this tiny local helper pins the
    template instead, without modifying questions.py."""
    template = question_generator.env.get_template(template_name)
    other_shapes = [s for s in question_generator.shapes if s != shape_type]
    context = {'shape': shape_type, 'random_other_shape': random.choice(other_shapes)}
    question, answer = template.render(**context).split('|', 1)
    return question.strip(), answer.strip()


def main():
    parser = argparse.ArgumentParser(description='Evaluate exact-match accuracy of the Toy VLM.')
    parser.add_argument('--checkpoint', type=str, default='toy_vlm.pth')
    parser.add_argument('--samples', type=int, default=200, help='Samples per template')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noise', action='store_true',
                         help='Evaluate on noisy images like training (default: clean images, like the GUI)')
    args = parser.parse_args()

    if args.samples < 1:
        parser.error('--samples must be >= 1')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    noise_desc = 'noisy' if args.noise else 'clean (noise-free)'
    condition_desc = 'matching training conditions' if args.noise else "matching the interactive GUI's default"
    print(
        "Note: training images include Gaussian noise; this evaluation uses "
        f"{noise_desc} images ({condition_desc}). "
        "Pass --noise to evaluate under training-like noisy conditions.\n"
    )

    # Load tokenizer + model via the shared loader (the checkpoint bundles
    # its vocab so they can never mismatch).
    model, tokenizer, _ = load_trained_model(args.checkpoint)
    model.to(DEVICE)
    print(f"Loaded checkpoint '{args.checkpoint}' onto device: {DEVICE}\n")

    shape_gen = ShapeGenerator()
    shapes = shape_gen.get_available_shapes()
    qgen = QuestionGenerator()

    families = {name: classify_family(qgen, name) for name in qgen.template_names}
    family_totals = {fam: Counter() for fam in FAMILIES}
    # Per-template gold-answer counts, used to compute each template's
    # template-memorization baseline (gold is a pure function of the
    # template, so a predictor that memorizes template->answer can score
    # up to the majority-gold rate within that template).
    template_golds = {name: Counter() for name in qgen.template_names}
    rows = []

    for name in qgen.template_names:
        fam = families[name]
        raw_question = (
            qgen.template_loader.templates_data[name]
            .split('|', 1)[0]
            .replace('{{ shape }}', '<shape>')
            .replace('{{ random_other_shape }}', '<other>')
            .strip()
        )
        correct = empty = exceptions = 0

        for _ in range(args.samples):
            shape_type = random.choice(shapes)
            image = shape_gen.generate_shape_image(shape_type, add_noise=args.noise)
            question, gold = render_for_template(qgen, name, shape_type)
            gold_norm = normalize(gold)
            template_golds[name][gold_norm] += 1

            had_exception = False
            try:
                pred = generate_response(model, image, question)
            except Exception:
                had_exception = True
                pred = ''

            pred_norm = normalize(pred)
            is_correct = pred_norm == gold_norm
            is_empty = pred_norm == ''

            correct += int(is_correct)
            empty += int(is_empty)
            exceptions += int(had_exception)

            family_totals[fam]['correct'] += int(is_correct)
            family_totals[fam]['total'] += 1
            family_totals[fam]['empty'] += int(is_empty)
            family_totals[fam]['exceptions'] += int(had_exception)

        template_baseline_count = template_golds[name].most_common(1)[0][1]
        rows.append((raw_question, fam, correct, args.samples, empty, exceptions, template_baseline_count))

    # --- Per-template table ---
    print(f"{'Template':40} {'Family':14} {'N':>5} {'Acc%':>7} {'Base%':>7} {'Empty':>6} {'Exc':>4}")
    print('-' * 90)
    for raw_question, fam, correct, n, empty, exceptions, baseline_count in rows:
        acc = 100.0 * correct / n
        base = 100.0 * baseline_count / n
        print(f"{raw_question[:40]:40} {fam:14} {n:>5} {acc:>6.1f}% {base:>6.1f}% {empty:>6} {exceptions:>4}")

    # --- Per-family rollups + template-memorization baseline ---
    # The family baseline is the sum, over the family's templates, of each
    # template's majority-gold count, divided by the family total -- i.e.
    # the accuracy achievable by memorizing template->answer, not a
    # majority vote pooled across templates (which understates it whenever
    # gold varies by template, as in the yes/no families).
    print('-' * 90)
    overall_correct = overall_total = overall_empty = overall_exceptions = 0
    for fam in FAMILIES:
        t = family_totals[fam]
        if t['total'] == 0:
            print(f"{fam:14} rollup:  skipped (no templates of this family in questions.txt)")
            continue
        acc = 100.0 * t['correct'] / t['total']
        fam_baseline_count = sum(
            template_golds[name].most_common(1)[0][1]
            for name in qgen.template_names if families[name] == fam
        )
        baseline = 100.0 * fam_baseline_count / t['total']
        print(
            f"{fam:14} rollup:  N={t['total']:>4}  acc={acc:6.1f}%  "
            f"template-memorization baseline={baseline:5.1f}%  empty={t['empty']:>4}  exceptions={t['exceptions']:>3}"
        )
        overall_correct += t['correct']
        overall_total += t['total']
        overall_empty += t['empty']
        overall_exceptions += t['exceptions']

    print('-' * 90)
    if overall_total == 0:
        print("OVERALL: no templates found (questions.txt appears to have no templates).")
    else:
        overall_acc = 100.0 * overall_correct / overall_total
        print(
            f"{'OVERALL':14}          N={overall_total:>4}  acc={overall_acc:6.1f}%  "
            f"empty={overall_empty:>4}  exceptions={overall_exceptions:>3}"
        )


if __name__ == '__main__':
    main()
