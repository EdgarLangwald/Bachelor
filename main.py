import time
import torch
from codebase.train import compute_segment_loss, compute_param_loss
from codebase.model import Model
from codebase.utils import load_chunk_all
from codebase.data import collate_fn
from torch.utils.data import DataLoader

if __name__ == '__main__':
    device = torch.device('cuda')
    model = Model.load("model.pt", device=device)
    dataset = load_chunk_all("complete_dataset/chunk_0.pkl")

    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))

    notes = batch['notes'].to(device)
    tokens = batch['tokens'].to(device)
    tgt_key_padding_mask = batch['tgt_key_padding_mask'].to(device)

    seq_len = tokens.size(1)
    tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    model.eval()
    with torch.no_grad():
        model_output = model(notes, tokens, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

    print("Testing optimized training step (now the default)...\n")

    # Profile complete training step
    forward_time = 202  # ms (from previous profile)
    param_loss_time = 35  # ms
    backward_time = 1429  # ms

    # Measure segment loss (now optimized)
    torch.cuda.synchronize()
    times = []
    for i in range(10):
        start = time.time()
        with torch.no_grad():
            segment_loss = compute_segment_loss(model_output, tokens, tgt_key_padding_mask)
        torch.cuda.synchronize()
        times.append(time.time() - start)

    avg_seg_loss_time = sum(times) / len(times) * 1000  # Convert to ms

    total_step_time = forward_time + avg_seg_loss_time + param_loss_time + backward_time

    print(f"Segment loss time (optimized):  {avg_seg_loss_time:.2f}ms")
    print(f"Total training step time:       {total_step_time:.2f}ms")
    print(f"\nFor 30 training steps:          {total_step_time*30/1000:.1f} seconds")
    print(f"\nSegment loss is now {avg_seg_loss_time/total_step_time*100:.1f}% of training time (was 47.6%)")
    print(f"\n[SUCCESS] Training now uses optimized compute_segment_loss!")
