import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np
from text import TextProcessor, MAX_SEQ_LEN, NUM_IMG_TOKENS, SimpleTokenizer
from utils_loss import compute_weighted_loss
from model import ToyVLM, generate_response
from shapes import ShapeGenerator, ObjType, ObjSize, SIZE_RANGES
from questions import RationaleGenerator
import os
import random
from runtime import setup_runtime


# Initialize runtime (device, optional compile helpers)
RUNTIME = setup_runtime()
DEVICE = RUNTIME["device"]
TO_DEVICE = RUNTIME["to_device"]
MAYBE_COMPILE = RUNTIME["maybe_compile"]


def build_examples(tp: TextProcessor, num_examples: int = 8) -> List[Dict]:
    sg = ShapeGenerator()
    examples: List[Dict] = []

    for _ in range(num_examples):
        img_np, meta = sg.generate_multi_shape_image(num_shapes=1, add_noise=True)
        # metadata for single shape
        m = meta[0]
        shape_present = m['shape']
        size_cat = m.get('size_category')
        img = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0) / 255.0

        # derive aux labels from metadata
        aux_counts = {e.value: 0 for e in ObjType}
        aux_counts[shape_present] = 1
        aux_size_counts = {e.value: 0 for e in ObjSize}
        if size_cat in aux_size_counts:
            aux_size_counts[size_cat] = 1

        # positive example (shape present)
        inp, tgt, rat_m, ans_m = tp.prepare_input_sequence(
            f"is there a {shape_present}", answer="yes", rationale=f"count {shape_present} is 1"
        )
        inp = tp.pad_sequence(inp, MAX_SEQ_LEN)
        tgt = tp.pad_sequence(tgt, MAX_SEQ_LEN)
        rat_m = tp.pad_sequence(rat_m, MAX_SEQ_LEN)
        ans_m = tp.pad_sequence(ans_m, MAX_SEQ_LEN)
        examples.append({
            'image': img,
            'input_ids': torch.tensor(inp, dtype=torch.long),
            'target_ids': torch.tensor(tgt, dtype=torch.long),
            'rat_mask': torch.tensor(rat_m, dtype=torch.float32),
            'ans_mask': torch.tensor(ans_m, dtype=torch.float32),
            'expected_answer': 'yes',
            'question': f"is there a {shape_present}",
            'aux_labels': {'counts': aux_counts, 'size_counts': aux_size_counts},
        })

        # negative example (different shape not present)
        present_enum = ObjType(shape_present)
        not_shape = random.choice([e for e in ObjType if e != present_enum])
        inp, tgt, rat_m, ans_m = tp.prepare_input_sequence(
            f"is there a {not_shape.value}", answer="no", rationale=f"count {not_shape.value} is 0"
        )
        inp = tp.pad_sequence(inp, MAX_SEQ_LEN)
        tgt = tp.pad_sequence(tgt, MAX_SEQ_LEN)
        rat_m = tp.pad_sequence(rat_m, MAX_SEQ_LEN)
        ans_m = tp.pad_sequence(ans_m, MAX_SEQ_LEN)
        examples.append({
            'image': img,
            'input_ids': torch.tensor(inp, dtype=torch.long),
            'target_ids': torch.tensor(tgt, dtype=torch.long),
            'rat_mask': torch.tensor(rat_m, dtype=torch.float32),
            'ans_mask': torch.tensor(ans_m, dtype=torch.float32),
            'expected_answer': 'no',
            'question': f"is there a {not_shape.value}",
            'aux_labels': {'counts': aux_counts, 'size_counts': aux_size_counts},
        })

    return examples


def overfit(model: ToyVLM, examples: List[Dict], iters: int = 200, lr: float = 3e-4) -> None:
    model.train().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tok = model.text_processor.tokenizer
    # Match train schedule for weights
    loss_weights = {'rationale': 1.0, 'answer': 2.0, 'aux': 0.5}

    for i in range(iters):
        total = 0.0
        total_mask_sum = 0.0
        sup_ce_all: List[torch.Tensor] = []
        for ex in examples:
            img = ex['image'].unsqueeze(0).to(DEVICE)
            inp = ex['input_ids'].unsqueeze(0).to(DEVICE)
            tgt = ex['target_ids'].unsqueeze(0).to(DEVICE)
            # Use masks from TextProcessor (supervises THINK and FINAL spans)
            msk = (ex['rat_mask'] + ex['ans_mask']).clamp(max=1).unsqueeze(0).to(DEVICE)

            # Structural assertions per-batch
            assert inp.size(1) == MAX_SEQ_LEN, "[test] Input must be padded to MAX_SEQ_LEN"
            # Run forward
            logits, aux_outputs = model(img, inp, return_aux=True)
            V = tok.get_vocab_size()
            # Use generated aux labels
            aux_labels = [ex['aux_labels']]
            loss, rat_loss, ans_loss, aux_loss = compute_weighted_loss(
                logits, tgt, ex['rat_mask'].unsqueeze(0).to(DEVICE), ex['ans_mask'].unsqueeze(0).to(DEVICE),
                aux_outputs, aux_labels, tok, loss_weights
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            total_mask_sum += float(msk.sum().item())
            # collect supervised CE values for diagnostics
            ce_full = F.cross_entropy(
                logits.reshape(-1, V),
                tgt.reshape(-1),
                ignore_index=tok.pad_token_id,
                reduction='none'
            ).view_as(tgt)
            sup_vals = ce_full[msk.bool()]
            if sup_vals.numel() > 0:
                sup_ce_all.append(sup_vals.detach().flatten())
        if (i+1) % 50 == 0:
            print(f"Iter {i+1}: loss={total/len(examples):.4f}")
        # Detailed diagnostics at early steps and every 25 iters
        if (i < 5) or ((i+1) % 25 == 0):
            if len(sup_ce_all) > 0:
                sup_all = torch.cat(sup_ce_all)
                sup_mean = float(sup_all.mean().item())
                sup_max = float(sup_all.max().item())
                sup_min = float(sup_all.min().item())
                head_vals = ", ".join([f"{v:.8f}" for v in sup_all[:6].tolist()])
                print(f"  diag: mask_sum={int(total_mask_sum)}, sup_ce_mean={sup_mean:.8f}, sup_ce_min={sup_min:.8f}, sup_ce_max={sup_max:.8f}, sup_ce_head=[{head_vals}]")
            else:
                print(f"  diag: mask_sum={int(total_mask_sum)}, no supervised tokens collected")


def verify_teacher_forcing(model: ToyVLM, examples: List[Dict]) -> bool:
    model.eval()
    tok = model.text_processor.tokenizer
    ok = True
    with torch.no_grad():
        for ex in examples:
            img = ex['image'].unsqueeze(0).to(DEVICE)
            inp = ex['input_ids'].unsqueeze(0).to(DEVICE)
            tgt = ex['target_ids'].unsqueeze(0).to(DEVICE)

            logits = model(img, inp)
            # Find index of <FINAL> in input (pre-shift). Target at same index is first answer token.
            final_idx = (inp[0] == tok.final_start_id).nonzero(as_tuple=True)[0]
            if final_idx.numel() == 0:
                ok = False
                continue
            k = int(final_idx[0].item())
            pred_id = int(logits[0, k, :].argmax().item())
            pred = tok.decode([pred_id], skip_special_tokens=True)
            exp = ex['expected_answer']
            print(f"Expected={exp}, Pred@TF={pred}")
            # Dump short window around FINAL for debugging
            window = list(range(max(0, k-5), min(int(inp.size(1)), k+6)))
            toks = [tok.decode([int(t)], skip_special_tokens=False) for t in inp[0, window]]
            print(f"Context around <FINAL>: {' '.join(toks)}")

            # Print all supervised tokens (expected vs actual)
            msk = (ex['rat_mask'] + ex['ans_mask']).clamp(max=1).unsqueeze(0).to(DEVICE)
            preds = logits.argmax(dim=-1)
            print("All supervised tokens (expected -> predicted):")
            mismatches = 0
            total_sup = 0
            for ti in range(tgt.size(1)):
                if int(msk[0, ti].item()) == 1:
                    total_sup += 1
                    exp_id = int(tgt[0, ti].item())
                    got_id = int(preds[0, ti].item())
                    exp_str = tok.decode([exp_id], skip_special_tokens=True) or f"<{exp_id}>"
                    got_str = tok.decode([got_id], skip_special_tokens=True) or f"<{got_id}>"
                    mark = "✓" if exp_id == got_id else "✗"
                    if exp_id != got_id:
                        mismatches += 1
                    print(f"  [{ti}] {exp_str} -> {got_str} {mark}")
            print(f"Supervised tokens: {total_sup}, mismatches: {mismatches}")
            ok = ok and (pred.strip() == exp.strip())
    return ok


def print_vision_token_diagnostics(model: ToyVLM, exs: List[Dict]) -> None:
    # Ensure model and tensors are on the same device
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        img_c = exs[0]['image'].unsqueeze(0).to(DEVICE)
        img_s = exs[2]['image'].unsqueeze(0).to(DEVICE)
        toks_c = model.encode_image_tokens(img_c)  # [1,N,H]
        toks_s = model.encode_image_tokens(img_s)  # [1,N,H]
        # Compare means and cosine directly over fixed NUM_IMG_TOKENS
        v_c = toks_c.mean(dim=1)  # [1,H]
        v_s = toks_s.mean(dim=1)  # [1,H]
        cos = torch.nn.functional.cosine_similarity(v_c, v_s).item()
        l2 = torch.norm(v_c - v_s).item()
        print(f"Vision diagnostics: pooled mean L2={l2:.4f}, cosine={cos:.4f}")


def build_dataset_examples(tp: TextProcessor, num_samples: int = 50, difficulty: str = 'easy', num_img_tokens: int = 64) -> List[Dict]:
    sg = ShapeGenerator()
    rg = RationaleGenerator()
    examples: List[Dict] = []
    for _ in range(num_samples):
        num_shapes = random.randint(1, 4)
        img_np, metadata = sg.generate_multi_shape_image(num_shapes, True)
        q, a, r = rg.generate_qa_with_rationale(metadata, difficulty=difficulty)
        inp, tgt, rat_m, ans_m = tp.prepare_input_sequence(q, a, r)
        inp = tp.pad_sequence(inp, MAX_SEQ_LEN)
        tgt = tp.pad_sequence(tgt, MAX_SEQ_LEN)
        rat_m = tp.pad_sequence(rat_m, MAX_SEQ_LEN)
        ans_m = tp.pad_sequence(ans_m, MAX_SEQ_LEN)
        img_t = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0) / 255.0
        examples.append({
            'image': img_t,
            'input_ids': torch.tensor(inp, dtype=torch.long),
            'target_ids': torch.tensor(tgt, dtype=torch.long),
            'rat_mask': torch.tensor(rat_m, dtype=torch.float32),
            'ans_mask': torch.tensor(ans_m, dtype=torch.float32),
            'question': q,
            'expected_answer': a,
            'rationale': r,
        })
    return examples


@torch.no_grad()
def verify_teacher_forcing_dataset(model: ToyVLM, examples: List[Dict], dump_failures: int = 3, topk: int = 8) -> float:
    model.eval()
    tok = model.text_processor.tokenizer
    correct = 0
    failures_shown = 0
    for ex in examples:
        img = ex['image'].unsqueeze(0).to(DEVICE)
        inp = ex['input_ids'].unsqueeze(0).to(DEVICE)
        tgt = ex['target_ids'].unsqueeze(0).to(DEVICE)
        logits = model(img, inp)
        final_idx = (inp[0] == tok.final_start_id).nonzero(as_tuple=True)[0]
        if final_idx.numel() == 0:
            continue
        k = int(final_idx[0].item())
        next_logits = logits[0, k, :]
        pred_id = int(next_logits.argmax().item())
        pred = tok.decode([pred_id], skip_special_tokens=True)
        exp = ex['expected_answer']
        if pred.strip() == exp.strip():
            correct += 1
        elif failures_shown < dump_failures:
            failures_shown += 1
            # dump top-k around FINAL
            scores, idxs = torch.topk(next_logits, k=topk)
            pairs = []
            for sc, ix in zip(scores.tolist(), idxs.tolist()):
                s = tok.decode([ix], skip_special_tokens=False) or f"<{ix}>"
                pairs.append(f"{s}:{sc:.2f}")
            print("\n[TF-DIAG] Failure example:")
            print(f"  Q: {ex['question']}")
            print(f"  GT: {exp}")
            print(f"  Pred@FINAL: {pred}")
            print(f"  Top-{topk} @FINAL: {', '.join(pairs)}")
    acc = correct / max(1, len(examples))
    print(f"TF accuracy over {len(examples)} examples: {acc:.2%}")
    return acc

@torch.no_grad()
def evaluate_auxiliary_heads(
    model: ToyVLM,
    num_samples: int = 120,
    batch_size: int = 16,
    add_noise: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate AuxiliaryHeads on multi-shape images.

    Returns:
        overall_metrics: dict with keys 'overall_acc', 'macro_acc', 'mae'
        per_head_acc: dict per shape/size head accuracy
    """
    model.eval().to(DEVICE)
    sg = ShapeGenerator()

    shape_keys = [e.value for e in ObjType]
    size_keys = [e.value for e in ObjSize]

    all_true: List[int] = []
    all_pred: List[int] = []
    per_head_correct = {k: 0 for k in (shape_keys + size_keys)}
    per_head_total = {k: 0 for k in (shape_keys + size_keys)}

    processed = 0
    while processed < num_samples:
        cur_bs = min(batch_size, num_samples - processed)
        imgs: List[torch.Tensor] = []
        # ground truth vectors per sample
        gt_shapes: List[List[int]] = []
        gt_sizes: List[List[int]] = []

        for _ in range(cur_bs):
            num_shapes = random.randint(2, 4)
            img_np, meta = sg.generate_multi_shape_image(num_shapes, add_noise)
            # grayscale (64,64) -> (1,64,64) in [0,1]
            img_t = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0) / 255.0
            imgs.append(img_t)

            # build ground-truth counts
            sc = {k: 0 for k in shape_keys}
            zc = {k: 0 for k in size_keys}
            for m in meta:
                shp = m['shape']
                siz = m['size_category']
                if shp in sc:
                    sc[shp] += 1
                if siz in zc:
                    zc[siz] += 1
            gt_shapes.append([sc[k] for k in shape_keys])
            gt_sizes.append([zc[k] for k in size_keys])

        batch = torch.stack(imgs, dim=0).to(DEVICE)  # [B,1,64,64]
        vtokens = model.encode_image_tokens(batch)   # [B,64,H]
        aux = model.auxiliary_heads(vtokens)

        # predictions: argmax over 0..4 classes
        pred_shapes = []
        for k in shape_keys:
            logits = aux['count_logits'][k]  # [B,5]
            pred = logits.argmax(dim=-1)
            pred_shapes.append(pred)
        pred_shapes_t = torch.stack(pred_shapes, dim=1)  # [B, num_shapes]

        pred_sizes = []
        for k in size_keys:
            logits = aux['size_count_logits'][k]  # [B,5]
            pred = logits.argmax(dim=-1)
            pred_sizes.append(pred)
        pred_sizes_t = torch.stack(pred_sizes, dim=1)  # [B, num_sizes]

        gt_shapes_t = torch.tensor(np.array(gt_shapes), device=DEVICE)
        gt_sizes_t = torch.tensor(np.array(gt_sizes), device=DEVICE)

        # accumulate overall stats
        all_true.append(gt_shapes_t.flatten())
        all_true.append(gt_sizes_t.flatten())
        all_pred.append(pred_shapes_t.flatten())
        all_pred.append(pred_sizes_t.flatten())

        # per-head accuracy
        for j, k in enumerate(shape_keys):
            correct = (pred_shapes_t[:, j] == gt_shapes_t[:, j]).sum().item()
            per_head_correct[k] += int(correct)
            per_head_total[k] += int(cur_bs)
        for j, k in enumerate(size_keys):
            correct = (pred_sizes_t[:, j] == gt_sizes_t[:, j]).sum().item()
            per_head_correct[k] += int(correct)
            per_head_total[k] += int(cur_bs)

        processed += cur_bs

    y_true = torch.cat(all_true).to(torch.float32)
    y_pred = torch.cat(all_pred).to(torch.float32)
    overall_acc = float((y_true == y_pred).float().mean().item())
    mae = float(torch.abs(y_true - y_pred).mean().item())

    # macro acc: mean of per-head accuracies
    head_accs = {k: (per_head_correct[k] / max(1, per_head_total[k])) for k in per_head_correct}
    macro_acc = float(np.mean(list(head_accs.values()))) if head_accs else 0.0

    overall_metrics = {
        'overall_acc': overall_acc,
        'macro_acc': macro_acc,
        'mae': mae,
        'num_samples': float(num_samples),
    }
    return overall_metrics, head_accs
 


def main():
    tp = TextProcessor()
    # Ensure required words
    words = ["is", "there", "a", "count", "0", "1", "yes", "no"] + [e.value for e in ObjType]
    for w in words:
        if w not in tp.tokenizer.vocab:
            tp.tokenizer.vocab[w] = len(tp.tokenizer.vocab)
    tp.tokenizer._update_mappings()

    model = ToyVLM(tp)
    model = TO_DEVICE(model)
    model = MAYBE_COMPILE(model)
    exs = build_examples(tp, num_examples=8)

    print("Overfitting 8-sample task...")
    print_vision_token_diagnostics(model, exs)
    overfit(model, exs, iters=200, lr=3e-4)

    print("\nVerifying teacher-forcing predictions...")
    ok = verify_teacher_forcing(model, exs)

    if ok:
        print("\nSUCCESS: Exact-match answers achieved under teacher forcing.")
    else:
        print("\nFAIL: Answers not matching under teacher forcing.")

    # Free-running generation check (question-only + image tokens)
    print("\nFree-running generation check:")
    for i, ex in enumerate(exs):
        # generate_response expects image tensor (C,H,W) in [0,1]
        img_chw = ex['image']  # already (1,64,64) float in [0,1]
        rat, ans = generate_response(model, img_chw, ex['question'], max_length=35, return_rationale=True)
        print(f"Sample {i}: expected={ex['expected_answer']}, generated answer='{ans}', rationale='{rat}'")

    # Evaluate AuxiliaryHeads on multi-shape images (count prediction)
    print("\nEvaluating AuxiliaryHeads (count prediction) on synthetic multi-shape images...")
    overall, per_head = evaluate_auxiliary_heads(model, num_samples=120, batch_size=16, add_noise=False)
    print(f"Aux Heads — overall_acc={overall['overall_acc']:.3f}, macro_acc={overall['macro_acc']:.3f}, MAE={overall['mae']:.3f}, samples={int(overall['num_samples'])}")
    # brief per-head breakdown
    def fmt_pair(k: str, v: float) -> str:
        return f"{k}:{v:.2f}"
    shape_keys = [e.value for e in ObjType]
    size_keys = [e.value for e in ObjSize]
    shape_line = ", ".join([fmt_pair(k, per_head.get(k, 0.0)) for k in shape_keys])
    size_line = ", ".join([fmt_pair(k, per_head.get(k, 0.0)) for k in size_keys])
    print(f"  Shapes acc — {shape_line}")
    print(f"  Sizes  acc — {size_line}")


if __name__ == "__main__":
    main()
