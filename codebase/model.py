import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
from .data import SegmentToken, Note


class NoteEmbedding(nn.Module):
    def __init__(self, d_model: int, mode: str = "concat"):
        super().__init__()
        self.d_model = d_model
        self.mode = mode

        if mode == "concat":
            assert d_model % 4 == 0, "d_model must be divisible by 12 for concat mode"
            embed_dim = d_model // 4
        else:
            embed_dim = d_model

        self.start_embed = nn.Linear(1, embed_dim)
        self.duration_embed = nn.Linear(1, embed_dim)
        self.pitch_embed = nn.Embedding(88, embed_dim)
        self.velocity_embed = nn.Embedding(17, embed_dim)

    def forward(self, notes):
        start = self.start_embed(notes[:, :, 0:1])
        duration = self.duration_embed(notes[:, :, 1:2])
        pitch = self.pitch_embed(notes[:, :, 2].long())
        velocity = self.velocity_embed(notes[:, :, 3].long())

        if self.mode == "concat":
            return torch.cat([start, duration, pitch, velocity], dim=-1)
        else:
            return start + duration + pitch + velocity


class SegmentEmbedding(nn.Module):
    def __init__(self, d_model: int, mode: str = "concat"):
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        self.sos_embed = nn.Parameter(torch.randn(d_model))

        if mode == "concat":
            assert d_model % 3 == 0, "d_model must be divisible by 12 for concat mode"
            embed_dim = d_model // 3
        else:
            embed_dim = d_model

        self.height_embed = nn.Linear(1, embed_dim)
        self.amount_embed = nn.Linear(1, embed_dim)
        self.time_embed = nn.Linear(1, embed_dim)
        self.first_amount_vector = nn.Parameter(torch.randn(embed_dim))

    def forward(self, tokens):
        batch_size, seq_len, _ = tokens.shape

        embeddings = torch.zeros(batch_size, seq_len, self.d_model, device=tokens.device)
        embeddings[:, 0] = self.sos_embed

        if seq_len > 1:
            h1 = self.height_embed(tokens[:, 1:2, 0].unsqueeze(-1)).squeeze(1)
            t1 = self.time_embed(tokens[:, 1:2, 2].unsqueeze(-1)).squeeze(1)
            if self.mode == "concat":
                embeddings[:, 1] = torch.cat([h1, self.first_amount_vector.unsqueeze(0).expand(batch_size, -1), t1], dim=-1)
            else:
                embeddings[:, 1] = h1 + self.first_amount_vector.unsqueeze(0) + t1

        if seq_len > 2:
            h = self.height_embed(tokens[:, 2:, 0].unsqueeze(-1))
            a = self.amount_embed(tokens[:, 2:, 1].unsqueeze(-1))
            t = self.time_embed(tokens[:, 2:, 2].unsqueeze(-1))
            if self.mode == "concat":
                embeddings[:, 2:] = torch.cat([h, a, t], dim=-1)
            else:
                embeddings[:, 2:] = h + a + t

        return embeddings


class Model(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        embedding_mode: str = "concat"
    ):
        super().__init__()

        self.config = {
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'num_decoder_layers': num_decoder_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'embedding_mode': embedding_mode
        }
        self.d_model = d_model

        self.note_embedding = NoteEmbedding(d_model, embedding_mode)
        self.segment_embedding = SegmentEmbedding(d_model, embedding_mode)
        self.note_pos_emb = nn.Embedding(600, d_model)
        self.seg_pos_emb = nn.Embedding(150, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.height_head = nn.Linear(d_model, 1)
        self.amount_head = nn.Linear(d_model, 1)
        self.time_head = nn.Linear(d_model, 1)

    def forward(self, notes, tokens, src_key_padding_mask=None, tgt_mask=None, tgt_key_padding_mask=None):
        note_emb = self.note_embedding(notes)
        batch_size, note_seq_len, _ = notes.shape
        note_positions = torch.arange(note_seq_len, device=notes.device).unsqueeze(0).expand(batch_size, -1)
        note_pos_emb = self.note_pos_emb(note_positions)
        note_emb = note_emb + note_pos_emb

        seg_emb = self.segment_embedding(tokens)
        _, seg_seq_len, _ = tokens.shape
        seg_positions = torch.arange(seg_seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        seg_pos_emb = self.seg_pos_emb(seg_positions)
        seg_emb = seg_emb + seg_pos_emb

        transformer_out = self.transformer(
            note_emb,
            seg_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        height_out = torch.sigmoid(4 * self.height_head(transformer_out) - 2)
        amount_out = torch.sigmoid(4 * self.amount_head(transformer_out) - 2)
        time_out = self.time_head(transformer_out)

        return {
            'height': height_out,
            'amount': amount_out,
            'time': time_out
        }

    def save(self, path: str = 'model.pt'):
        import torch
        from pathlib import Path
        import contextlib, io

        path = f'saves/{path}'
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'config': self.config,
            'state_dict': self.state_dict()
        }

        with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
            torch.save(checkpoint, path)

        return None

    @classmethod
    def load(cls, path: str = 'model.pt', device: str = 'cpu'):
        import torch
        path = f'saves/{path}'
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        return model
