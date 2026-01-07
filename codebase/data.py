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
        return notes[-1].start + notes[-1].duration

    def __call__(self, track_idx: int = None, time: float = None, window_size=10) -> Tuple[List[Note], List[SegmentToken]]:
        from .preprocessing import segment

        if track_idx is None:
            track_idx = random.randint(0, len(self.tracks) - 1)

        track_length = self.get_track_length(track_idx)

        if time is not None:
            assert time <= track_length - window_size, f"Choose time between 0 and {track_length - window_size} seconds"
        else:
            time = random.uniform(0, track_length - window_size)

        notes, tokens = self.tracks[track_idx]
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

        windowed_tokens = [
            SegmentToken(
                height=s.height,
                amount=s.amount,
                time=s.time - time
            )
            for s in tokens if time < s.time < window_end
        ]

        sos = SegmentToken(height=0.0, amount=0.0, time=0.0)
        result_tokens = [sos]

        if tokens[0].time <= time and tokens[-1].time > time:
            left_border_token = self._create_border_token(tokens, time, window_size, is_left=True)
            result_tokens.append(left_border_token)

        result_tokens.extend(windowed_tokens)

        if tokens[0].time <= window_end and tokens[-1].time > window_end:
            right_border_token = self._create_border_token(tokens, time, window_size, is_left=False)
            result_tokens.append(right_border_token)

        windowed_tokens = result_tokens

        return windowed_notes, windowed_tokens

    def _create_border_token(self, tokens: List[SegmentToken], time: float, window_size: float, is_left: bool) -> SegmentToken:
        from .preprocessing import segment

        border_time = time if is_left else time + window_size

        left_token = None
        right_token = None
        for i, token in enumerate(tokens):
            if token.time <= border_time:
                left_token = token
            if token.time > border_time and right_token is None:
                right_token = token
                break

        assert left_token is not None and right_token is not None, "Border tokens not found"

        seg = segment(left_token.time, left_token.height, right_token.time, right_token.height, right_token.amount)
        border_height = seg(border_time)

        if is_left:
            return SegmentToken(height=border_height, amount=0.0, time=0.0)
        else:
            return SegmentToken(height=border_height, amount=right_token.amount, time=window_size)

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
