import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from typing import List, Tuple
from codebase.data import Note, SegmentToken
from codebase.inference import generate_batch, tokens_to_segs
from codebase.utils import segs_to_curve, load_dataset
from codebase.model import Model
from codebase.train import compute_segment_loss, compute_param_loss


def correlation(gt_values: np.ndarray, pred_values: np.ndarray) -> float:
    if len(gt_values) == 0 or len(pred_values) == 0:
        return np.nan
    min_len = min(len(gt_values), len(pred_values))
    gt_values, pred_values = gt_values[:min_len], pred_values[:min_len]
    if np.std(gt_values) == 0 or np.std(pred_values) == 0:
        return np.nan
    return np.corrcoef(gt_values, pred_values)[0, 1]


def binary_metrics(gt_values: np.ndarray, pred_values: np.ndarray, threshold: float = 0.5) -> dict:
    min_len = min(len(gt_values), len(pred_values))
    gt_binary = (gt_values[:min_len] >= threshold).astype(float)
    pred_binary = (pred_values[:min_len] >= threshold).astype(float)

    tp = np.sum((gt_binary == 1) & (pred_binary == 1))
    tn = np.sum((gt_binary == 0) & (pred_binary == 0))
    fp = np.sum((gt_binary == 0) & (pred_binary == 1))
    fn = np.sum((gt_binary == 1) & (pred_binary == 0))

    binary_corr = correlation(gt_binary, pred_binary)

    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'binary_correlation': binary_corr
    }


def num_tokens_ratio(gt_tokens: List[SegmentToken], pred_tokens: List[SegmentToken]) -> float:
    return (len(pred_tokens) - 1) / (len(gt_tokens) - 1)


def compute_batch_loss(model, notes_batch: List[List[Note]], gt_tokens_batch: List[List[SegmentToken]], device: str) -> dict:
    batch_size = len(notes_batch)
    max_notes = max(len(notes) for notes in notes_batch)
    max_tokens = max(len(tokens) for tokens in gt_tokens_batch)

    notes_tensor = torch.zeros(batch_size, max_notes, 4, dtype=torch.float32, device=device)
    tokens_tensor = torch.zeros(batch_size, max_tokens, 3, dtype=torch.float32, device=device)
    src_key_padding_mask = torch.ones(batch_size, max_notes, dtype=torch.bool, device=device)
    tgt_key_padding_mask = torch.ones(batch_size, max_tokens, dtype=torch.bool, device=device)

    for b, (notes, tokens) in enumerate(zip(notes_batch, gt_tokens_batch)):
        for i, note in enumerate(notes):
            notes_tensor[b, i] = torch.tensor([note.start, note.duration, note.pitch, note.velocity])
        src_key_padding_mask[b, :len(notes)] = False
        for i, tok in enumerate(tokens):
            tokens_tensor[b, i] = torch.tensor([tok.height, tok.amount, tok.time])
        tgt_key_padding_mask[b, :len(tokens)] = False

    tgt_mask = torch.triu(torch.ones(max_tokens, max_tokens, device=device), diagonal=1).bool()
    model_output = model(notes_tensor, tokens_tensor, src_key_padding_mask=src_key_padding_mask, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

    return {
        'segment': compute_segment_loss(model_output, tokens_tensor, tgt_key_padding_mask).item(),
        'param': compute_param_loss(model_output, tokens_tensor, tgt_key_padding_mask).item()
    }


@torch.no_grad()
def evaluate(
    model,
    dataset,
    num_samples: int = 100,
    device: str = 'cpu',
    threshold: float = 0.5,
    seed: int = None,
    batch_size: int = 32
) -> dict:
    import random

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    samples = []
    while len(samples) < num_samples:
        notes, tokens = dataset()
        if len(tokens) >= 3:
            samples.append((notes, tokens))

    results = {'correlation': [], 'num_tokens_ratio': [], 'binary_correlation': []}
    binary_totals = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    loss_totals = {'segment': 0.0, 'param': 0.0}
    loss_count = 0

    model.eval()

    for batch_start in range(0, num_samples, batch_size):
        batch_samples = samples[batch_start:batch_start + batch_size]
        notes_batch = [s[0] for s in batch_samples]
        gt_tokens_batch = [s[1] for s in batch_samples]
        curr_batch_size = len(batch_samples)

        pred_tokens_batch = generate_batch(model, notes_batch, max_length=127, device=device)

        batch_loss = compute_batch_loss(model, notes_batch, gt_tokens_batch, device)
        loss_totals['segment'] += batch_loss['segment'] * curr_batch_size
        loss_totals['param'] += batch_loss['param'] * curr_batch_size
        loss_count += curr_batch_size

        for gt_tokens, pred_tokens in zip(gt_tokens_batch, pred_tokens_batch):
            gt_segs = tokens_to_segs(gt_tokens)
            pred_segs = tokens_to_segs(pred_tokens)

            _, gt_values = segs_to_curve(gt_segs, num_points=1000)
            _, pred_values = segs_to_curve(pred_segs, num_points=1000)

            results['correlation'].append(correlation(gt_values, pred_values))
            results['num_tokens_ratio'].append(num_tokens_ratio(gt_tokens, pred_tokens))

            binary = binary_metrics(gt_values, pred_values, threshold)
            results['binary_correlation'].append(binary['binary_correlation'])
            for key in binary_totals:
                binary_totals[key] += binary[key]

    tp, tn, fp, fn = binary_totals['tp'], binary_totals['tn'], binary_totals['fp'], binary_totals['fn']
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    return {
        'correlation': np.nanmean(results['correlation']),
        'binary_correlation': np.nanmean(results['binary_correlation']),
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1,
        'num_tokens_ratio': np.mean(results['num_tokens_ratio']),
        'segment_loss': loss_totals['segment'] / loss_count,
        'param_loss': loss_totals['param'] / loss_count
    }

if __name__ == '__main__':
    model = Model.load("HPC_model.pt", device="cuda")
    dataset = load_dataset("test_set/chunk_0.pkl")

    lines = [
        "# HPC Model Evaluation Results\n",
        "| Alpha | Ps. Corr | Bin. Corr | Prec. | Rec. | Acc. |  F1  | Tok. Ratio | Seg Loss | Param Loss |",
        "|-------|----------|-----------|-------|------|------|------|------------|----------|------------|"
    ]

    for alpha in [0.2, 0.35, 0.5, 0.65, 0.8]:
        r = evaluate(model, dataset, 100, "cuda", alpha, seed=0, batch_size=50)
        row = f"| {alpha:.2f}  | {r['correlation']:>8.2f} | {r['binary_correlation']:>9.2f} | {r['precision']:>5.2f} | {r['recall']:>4.2f} | {r['accuracy']:>4.2f} | {r['f1']:>4.2f} | {r['num_tokens_ratio']:>10.2f} | {r['segment_loss']:>8.2f} | {r['param_loss']:>10.2f} |"
        lines.append(row)
        print(row)

    with open("saves/HPC_eval.md", "w") as f:
        f.write("\n".join(lines))