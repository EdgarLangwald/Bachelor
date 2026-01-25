import sys
import os
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import torch
import random
from codebase.model import Model
from codebase.data import collate_fn
from codebase.train import compute_segment_loss, compute_param_loss
from codebase.evaluate import evaluate
from codebase.utils import load_dataset

device = torch.device('cuda')
dataset = load_dataset("400Mb/chunk_0.pkl")
num_batches = 50
batch_size = 50
num_eval_samples = 100

print(f"Evaluating on {num_batches} batches × {batch_size} = {num_batches * batch_size} samples for loss")
print(f"Evaluating on {num_eval_samples} samples for metrics\n")

results = []

for alpha in [0, 0.25, 0.5, 0.75, 1]:
    random.seed(42)
    model = Model.load(f"alpha_{alpha}.pt", device)
    model.eval()

    seg_total = 0.0
    param_total = 0.0

    with torch.no_grad():
        for _ in range(num_batches):
            samples = []
            while len(samples) < batch_size:
                sample = dataset()
                if len(sample[0]) > 0 and len(sample[1]) >= 2:
                    samples.append(sample)

            batch = collate_fn(samples)
            notes = batch['notes'].to(device)
            tokens = batch['tokens'].to(device)
            src_key_padding_mask = batch['src_key_padding_mask'].to(device)
            tgt_key_padding_mask = batch['tgt_key_padding_mask'].to(device)

            seq_len = tokens.size(1)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

            model_output = model(
                notes, tokens,
                src_key_padding_mask=src_key_padding_mask,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

            seg_total += compute_segment_loss(model_output, tokens, tgt_key_padding_mask).item()
            param_total += compute_param_loss(model_output, tokens, tgt_key_padding_mask).item()

    seg_loss = seg_total / num_batches
    param_loss = param_total / num_batches
    total_loss = seg_loss + param_loss

    random.seed(42)
    metrics = evaluate(model, dataset, num_samples=num_eval_samples, device=device, seed=42)

    results.append({
        'alpha': alpha,
        'correlation': metrics['correlation'],
        'binary_corr': metrics['binary_correlation'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'tokens_ratio': metrics['num_tokens_ratio'],
        'seg_loss': seg_loss,
        'param_loss': param_loss,
        'total_loss': total_loss
    })

lines = [
    "# Model Evaluation Results\n",
    "| Alpha | Correlation | Binary Corr | Precision | Recall | Accuracy |   F1   | Tokens Ratio | Seg Loss | Param Loss | Total Loss |",
    "|-------|-------------|-------------|-----------|--------|----------|--------|--------------|----------|------------|------------|"
]
for r in results:
    lines.append(f"| {r['alpha']:<5} | {r['correlation']:>11.4f} | {r['binary_corr']:>11.4f} | {r['precision']:>9.4f} | {r['recall']:>6.4f} | {r['accuracy']:>8.4f} | {r['f1']:>6.4f} | {r['tokens_ratio']:>12.4f} | {r['seg_loss']:>8.4f} | {r['param_loss']:>10.4f} | {r['total_loss']:>10.4f} |")

output = "\n".join(lines) + "\n"
with open("evaluation_results.md", "w") as f:
    f.write(output)
print(output)
print("Written to evaluation_results.md")
