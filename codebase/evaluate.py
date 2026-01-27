import torch
import numpy as np
from typing import List, Tuple
from .data import Note, SegmentToken, collate_fn
from .inference import generate, tokens_to_segs
from .utils import segs_to_curve
from .train import compute_segment_loss, compute_param_loss


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


def evaluate_single(
    model,
    notes: List[Note],
    gt_tokens: List[SegmentToken],
    device: str = 'cpu',
    threshold: float = 0.5,
    num_points: int = 1000
) -> dict:
    pred_tokens = generate(model, notes, max_length=127, device=device)

    gt_segs = tokens_to_segs(gt_tokens)
    pred_segs = tokens_to_segs(pred_tokens)

    gt_times, gt_values = segs_to_curve(gt_segs, num_points=num_points)
    pred_times, pred_values = segs_to_curve(pred_segs, num_points=num_points)

    return {
        'correlation': correlation(gt_values, pred_values),
        'binary': binary_metrics(gt_values, pred_values, threshold),
        'num_tokens_ratio': num_tokens_ratio(gt_tokens, pred_tokens)
    }


def evaluate(
    model,
    dataset,
    num_samples: int = 100,
    device: str = 'cpu',
    threshold: float = 0.5,
    seed: int = None,
    save: str = None,
    batch_size: int = 50
) -> dict:
    import random

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    samples = [dataset() for _ in range(num_samples)]

    results = {'correlation': [], 'num_tokens_ratio': [], 'binary_correlation': []}
    binary_totals = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

    for notes, tokens in samples:
        sample_results = evaluate_single(model, notes, tokens, device, threshold)
        results['correlation'].append(sample_results['correlation'])
        results['num_tokens_ratio'].append(sample_results['num_tokens_ratio'])
        results['binary_correlation'].append(sample_results['binary']['binary_correlation'])
        for key in binary_totals:
            binary_totals[key] += sample_results['binary'][key]

    seg_total, param_total = 0.0, 0.0
    num_batches = (num_samples + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(num_batches):
            batch_samples = samples[i * batch_size : (i + 1) * batch_size]
            batch = collate_fn(batch_samples)
            notes_batch = batch['notes'].to(device)
            tokens_batch = batch['tokens'].to(device)
            src_key_padding_mask = batch['src_key_padding_mask'].to(device)
            tgt_key_padding_mask = batch['tgt_key_padding_mask'].to(device)

            seq_len = tokens_batch.size(1)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

            model_output = model(
                notes_batch, tokens_batch,
                src_key_padding_mask=src_key_padding_mask,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            seg_total += compute_segment_loss(model_output, tokens_batch, tgt_key_padding_mask).item()
            param_total += compute_param_loss(model_output, tokens_batch, tgt_key_padding_mask).item()

    seg_loss = seg_total / num_batches
    param_loss = param_total / num_batches

    tp, tn, fp, fn = binary_totals['tp'], binary_totals['tn'], binary_totals['fp'], binary_totals['fn']
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    final_results = {
        'correlation': np.nanmean(results['correlation']),
        'binary_correlation': np.nanmean(results['binary_correlation']),
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'f1': f1,
        'num_tokens_ratio': np.mean(results['num_tokens_ratio']),
        'seg_loss': seg_loss,
        'param_loss': param_loss,
        'total_loss': seg_loss + param_loss
    }

    if save is not None:
        r = final_results
        lines = [
            "# Evaluation Results\n",
            "| Ps. Corr | Bin. Corr | Prec. | Rec. | Acc. |   F1   | Tok. Ratio | Seg Loss | Param Loss | Total Loss |",
            "|----------|-----------|-------|------|------|--------|------------|----------|------------|------------|",
            f"|   {r['correlation']:.2f}   |   {r['binary_correlation']:.2f}    | {r['precision']:.2f}  | {r['recall']:.2f} | {r['accuracy']:.2f} |  {r['f1']:.2f}  |    {r['num_tokens_ratio']:.2f}    |   {r['seg_loss']:.2f}   |    {r['param_loss']:.2f}    |    {r['total_loss']:.2f}    |"
        ]
        with open("saves/" + save, "w") as f:
            f.write("\n".join(lines) + "\n")

    return final_results
