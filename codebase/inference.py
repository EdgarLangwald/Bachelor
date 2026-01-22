import torch
from typing import List
from .data import SegmentToken, SegmentEvent, Note


@torch.no_grad
def generate(model, notes: List[Note], max_length: int = 127, device: str = 'cpu') -> List[SegmentToken]:
    model.eval()

    notes_array = [[note.start, note.duration, note.pitch, note.velocity] for note in notes]
    notes_tensor = torch.tensor(notes_array, dtype=torch.float32, device=device).reshape(1, len(notes), 4)

    generated_tokens = [SegmentToken(height=0.0, amount=0.0, time=0.0)]

    for step in range(max_length):
        tgt_len = len(generated_tokens)
        tokens_tensor = torch.zeros(1, tgt_len, 3, dtype=torch.float32, device=device)
        for i, token in enumerate(generated_tokens):
            tokens_tensor[0, i] = torch.tensor([token.height, token.amount, token.time])

        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()
        model_output = model(notes_tensor, tokens_tensor, tgt_mask=tgt_mask)

        last_idx = tgt_len - 1

        height = model_output['height'][0, last_idx, 0].item()
        amount = model_output['amount'][0, last_idx, 0].item()

        log_time_delta = model_output['time'][0, last_idx, 0]
        time_delta = torch.exp(log_time_delta).item()
        cumulative_time = generated_tokens[-1].time + time_delta

        generated_tokens.append(SegmentToken(height=height, amount=amount, time=cumulative_time))

        if generated_tokens[-1].time >= 10:
            break

    return generated_tokens


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
