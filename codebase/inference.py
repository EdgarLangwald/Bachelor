import torch
from typing import List
from .data import SegmentToken, SegmentEvent, Note


@torch.no_grad()
def generate_batch(model, notes_batch: List[List[Note]], max_length: int = 127, device: str = 'cpu') -> List[List[SegmentToken]]:
    """Batched generation - processes multiple samples in parallel on GPU."""
    model.eval()
    batch_size = len(notes_batch)

    max_notes = max(len(notes) for notes in notes_batch)
    notes_tensor = torch.zeros(batch_size, max_notes, 4, dtype=torch.float32, device=device)
    notes_mask = torch.ones(batch_size, max_notes, dtype=torch.bool, device=device)

    for b, notes in enumerate(notes_batch):
        for i, note in enumerate(notes):
            notes_tensor[b, i] = torch.tensor([note.start, note.duration, note.pitch, note.velocity])
        notes_mask[b, :len(notes)] = False

    tokens_tensor = torch.zeros(batch_size, max_length + 1, 3, dtype=torch.float32, device=device)
    active = torch.ones(batch_size, dtype=torch.bool, device=device)
    seq_lens = torch.ones(batch_size, dtype=torch.long, device=device)

    for step in range(max_length):
        if not active.any():
            break

        tgt_len = step + 1
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()

        model_output = model(
            notes_tensor,
            tokens_tensor[:, :tgt_len],
            tgt_mask=tgt_mask,
            src_key_padding_mask=notes_mask
        )

        heights = model_output['height'][:, -1, 0]
        amounts = model_output['amount'][:, -1, 0]
        time_deltas = torch.exp(model_output['time'][:, -1, 0])
        new_times = tokens_tensor[:, step, 2] + time_deltas

        tokens_tensor[:, step + 1, 0] = heights
        tokens_tensor[:, step + 1, 1] = amounts
        tokens_tensor[:, step + 1, 2] = new_times

        finished = new_times >= 9.9
        seq_lens = torch.where(active & ~finished, seq_lens + 1, seq_lens)
        seq_lens = torch.where(active & finished, seq_lens + 1, seq_lens)
        active = active & ~finished

    results = []
    tokens_cpu = tokens_tensor.cpu()
    seq_lens_cpu = seq_lens.cpu()
    for b in range(batch_size):
        sample_tokens = [SegmentToken(height=0.0, amount=0.0, time=0.0)]
        for i in range(1, seq_lens_cpu[b].item() + 1):
            sample_tokens.append(SegmentToken(
                height=tokens_cpu[b, i, 0].item(),
                amount=tokens_cpu[b, i, 1].item(),
                time=tokens_cpu[b, i, 2].item()
            ))
        results.append(sample_tokens)

    return results


@torch.no_grad()
def generate(model, notes: List[Note], max_length: int = 127, device: str = 'cpu') -> List[SegmentToken]:
    return generate_batch(model, [notes], max_length, device)[0]


def tokens_to_segs(tokens: List[SegmentToken]) -> List[SegmentEvent]:
    if len(tokens) < 3:
        return []

    segments = []
    segment_tokens = tokens[1:]

    first, second = segment_tokens[0], segment_tokens[1]
    segments.append(SegmentEvent(
        x_start=first.time,
        y_start=first.height,
        x_end=second.time,
        y_end=second.height,
        amount=second.amount
    ))
    
    for token in segment_tokens[2:]:
        prev = segments[-1]
        segments.append(SegmentEvent(
            x_start=prev.x_end,
            y_start=prev.y_end,
            x_end=token.time,
            y_end=token.height,
            amount=token.amount
        ))

    return segments
