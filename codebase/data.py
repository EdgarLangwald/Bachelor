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
            for s in tokens if time <= s.time <= window_end
        ]
        sos = SegmentToken(height=0.0, amount=0.0, time=0.0)
        eos = SegmentToken(height=0.0, amount=0.0, time=0.0)
        windowed_tokens = [sos] + windowed_tokens + [eos]

        return windowed_notes, windowed_tokens

    def __getitem__(self, idx: int) -> Tuple[List[Note], List[SegmentToken]]:
        return self()

    def __len__(self) -> int:
        return len(self.tracks)


def collate_fn(batch: List[Tuple[List[Note], List[SegmentToken]]]) -> Dict[str, torch.Tensor]:
    batch_notes = []
    batch_tokens = []

    for notes, tokens in batch:
        batch_notes.append(notes)
        batch_tokens.append(tokens)

    max_note_len = max(len(notes) for notes in batch_notes)
    max_token_len = max(len(tokens) for tokens in batch_tokens)
    batch_size = len(batch)

    notes_data = np.zeros((batch_size, max_note_len, 4), dtype=np.float32)
    src_key_padding_mask = np.ones((batch_size, max_note_len), dtype=bool)

    for i, notes in enumerate(batch_notes):
        note_len = len(notes)
        for j, note in enumerate(notes):
            notes_data[i, j] = [note.start, note.duration, note.pitch, note.velocity]
        src_key_padding_mask[i, :note_len] = False

    tokens_data = np.zeros((batch_size, max_token_len, 3), dtype=np.float32)
    tgt_key_padding_mask = np.ones((batch_size, max_token_len), dtype=bool)

    for i, tokens in enumerate(batch_tokens):
        token_len = len(tokens)
        for j, token in enumerate(tokens):
            tokens_data[i, j] = [token.height, token.amount, token.time]
        tgt_key_padding_mask[i, :token_len] = False

    return {
        'notes': torch.from_numpy(notes_data),
        'tokens': torch.from_numpy(tokens_data),
        'src_key_padding_mask': torch.from_numpy(src_key_padding_mask),
        'tgt_key_padding_mask': torch.from_numpy(tgt_key_padding_mask),
    }
