from dataclasses import dataclass
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset as TorchDataset
import random
import numpy as np


@dataclass
class PedalEvent:
    time: float
    value: float


@dataclass
class Note:
    start: float
    duration: float
    pitch: int
    velocity: int


@dataclass
class SegmentEvent:
    x_start: float
    y_start: float
    x_end: float
    y_end: float
    amount: float


@dataclass
class SegmentToken:
    height: float
    amount: float
    time: float


class Dataset(TorchDataset):
    def __init__(self):
        self.tracks = [] # type(track) = Tuple[List[Note], List[SegmentToken]]

    def add_tracks(self, notes: List[Note], tokens: List[SegmentToken]):
        self.tracks.append((notes, tokens))

    def get_track_length(self, track_idx: int) -> float:
        notes, tokens = self.tracks[track_idx]
        return max(notes[-1].start + notes[-1].duration, tokens[-1].time)

    def __call__(self, track_idx: int = None, time: float = None, window_size=10) -> Tuple[List[Note], List[SegmentToken]]:
        import bisect
        from .preprocessing import segment

        if track_idx is None:
            track_idx = random.randint(0, len(self.tracks) - 1)

        notes, tokens = self.tracks[track_idx]
        track_length = self.get_track_length(track_idx)
        min_time = tokens[0].time

        if time is not None:
            assert min_time <= time <= track_length - window_size, f"Choose time between {min_time} and {track_length - window_size} seconds"
        else:
            time = random.uniform(min_time, track_length - window_size)
        window_end = time + window_size

        windowed_notes = [
            Note(
                start=n.start - time,
                duration=n.duration,
                pitch=n.pitch,
                velocity=n.velocity
            )
            for n in notes
            if n.start < window_end and n.start + n.duration > time
        ]

        token_times = [t.time for t in tokens]

        lr_idx = bisect.bisect_right(token_times, time)
        ll_idx = lr_idx - 1
        rr_idx = bisect.bisect_right(token_times, window_end)
        rl_idx = rr_idx - 1

        result_tokens = [SegmentToken(height=0.0, amount=0.0, time=0.0)]

        if lr_idx >= len(tokens):
            left_border = SegmentToken(height=tokens[ll_idx].height, amount=0.0, time=0.0)
            inner_start_idx = len(tokens)
        elif tokens[lr_idx].time - time < 0.1:
            left_border = SegmentToken(height=tokens[lr_idx].height, amount=0.0, time=0.0)
            inner_start_idx = lr_idx + 1
        else:
            seg = segment(tokens[ll_idx].time, tokens[ll_idx].height, tokens[lr_idx].time, tokens[lr_idx].height, tokens[lr_idx].amount)
            left_border = SegmentToken(height=seg(time), amount=0.0, time=0.0)
            inner_start_idx = lr_idx
        result_tokens.append(left_border)

        for i in range(inner_start_idx, rl_idx + 1):
            result_tokens.append(SegmentToken(
                height=tokens[i].height,
                amount=tokens[i].amount,
                time=tokens[i].time - time
            ))

        if rr_idx >= len(tokens):
            right_border = SegmentToken(height=tokens[rl_idx].height, amount=0.5, time=window_size)
        elif window_end - tokens[rl_idx].time < 0.1 and len(result_tokens) > 2:
            result_tokens[-1] = SegmentToken(height=tokens[rl_idx].height, amount=tokens[rl_idx].amount, time=window_size)
            right_border = None
        else:
            seg = segment(tokens[rl_idx].time, tokens[rl_idx].height, tokens[rr_idx].time, tokens[rr_idx].height, tokens[rr_idx].amount)
            right_border = SegmentToken(height=seg(window_end), amount=tokens[rr_idx].amount, time=window_size)

        if right_border is not None:
            result_tokens.append(right_border)

        return windowed_notes, result_tokens

    def __getitem__(self, idx: int) -> Tuple[List[Note], List[SegmentToken]]:
        return self()

    def __len__(self) -> int:
        return len(self.tracks)


def collate_fn(batch: List[Tuple[List[Note], List[SegmentToken]]]) -> Dict[str, torch.Tensor]:
    MAX_NOTES = 600
    MAX_TOKENS = 150

    batch_notes = [notes[:MAX_NOTES] for notes, _ in batch]
    batch_tokens = [tokens[:MAX_TOKENS] for _, tokens in batch]

    max_note_len = max(len(notes) for notes in batch_notes)
    max_token_len = max(len(tokens) for tokens in batch_tokens)
    batch_size = len(batch)

    notes_data = torch.zeros((batch_size, max_note_len, 4), dtype=torch.float32)
    src_key_padding_mask = torch.ones((batch_size, max_note_len), dtype=torch.bool)

    for i, notes in enumerate(batch_notes):
        note_len = len(notes)
        if note_len > 0:
            notes_array = np.array([[n.start, n.duration, n.pitch, n.velocity] for n in notes], dtype=np.float32)
            notes_data[i, :note_len] = torch.from_numpy(notes_array)
            src_key_padding_mask[i, :note_len] = False

    tokens_data = torch.zeros((batch_size, max_token_len, 3), dtype=torch.float32)
    tgt_key_padding_mask = torch.ones((batch_size, max_token_len), dtype=torch.bool)

    for i, tokens in enumerate(batch_tokens):
        token_len = len(tokens)
        if token_len > 0:
            tokens_array = np.array([[t.height, t.amount, t.time] for t in tokens], dtype=np.float32)
            tokens_data[i, :token_len] = torch.from_numpy(tokens_array)
            tgt_key_padding_mask[i, :token_len] = False

    return {
        'notes': notes_data,
        'tokens': tokens_data,
        'src_key_padding_mask': src_key_padding_mask,
        'tgt_key_padding_mask': tgt_key_padding_mask,
    }
