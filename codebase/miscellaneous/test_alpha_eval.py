import torch
import random
from codebase.model import Model
from codebase.data import collate_fn
from codebase.train import compute_segment_loss, compute_param_loss
from codebase.utils import load_dataset

device = torch.device('cuda')
dataset = load_dataset("400Mb/chunk_0.pkl")
num_batches = 50
batch_size = 50

print(f"Evaluating on {num_batches} batches × {batch_size} = {num_batches * batch_size} samples...\n")
print(f"{'Alpha':<10} {'Segment Loss':<15} {'Param Loss':<15}")
print("-" * 40)

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

    print(f"{alpha:<10} {seg_total/num_batches:<15.4f} {param_total/num_batches:<15.4f}")
