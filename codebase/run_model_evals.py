import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from codebase.evaluate import evaluate
from codebase.utils import load_dataset


# Old model classes that support embedding_mode for loading legacy models
class NoteEmbeddingLegacy(nn.Module):
    def __init__(self, d_model: int, mode: str = "concat"):
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        embed_dim = d_model // 4 if mode == "concat" else d_model
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


class SegmentEmbeddingLegacy(nn.Module):
    def __init__(self, d_model: int, mode: str = "concat"):
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        self.sos_embed = nn.Parameter(torch.randn(d_model))
        embed_dim = d_model // 3 if mode == "concat" else d_model
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


class ModelLegacy(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout=0.1, embedding_mode="concat"):
        super().__init__()
        self.d_model = d_model
        self.note_embedding = NoteEmbeddingLegacy(d_model, embedding_mode)
        self.segment_embedding = SegmentEmbeddingLegacy(d_model, embedding_mode)
        self.note_pos_emb = nn.Embedding(600, d_model)
        self.seg_pos_emb = nn.Embedding(150, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.height_head = nn.Linear(d_model, 1)
        self.amount_head = nn.Linear(d_model, 1)
        self.time_head = nn.Linear(d_model, 1)

    def forward(self, notes, tokens, src_key_padding_mask=None, tgt_mask=None, tgt_key_padding_mask=None):
        note_emb = self.note_embedding(notes)
        batch_size, note_seq_len, _ = notes.shape
        note_positions = torch.arange(note_seq_len, device=notes.device).unsqueeze(0).expand(batch_size, -1)
        note_emb = note_emb + self.note_pos_emb(note_positions)
        seg_emb = self.segment_embedding(tokens)
        _, seg_seq_len, _ = tokens.shape
        seg_positions = torch.arange(seg_seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        seg_emb = seg_emb + self.seg_pos_emb(seg_positions)
        transformer_out = self.transformer(
            note_emb, seg_emb, src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        return {
            'height': torch.sigmoid(4 * self.height_head(transformer_out) - 2),
            'amount': torch.sigmoid(4 * self.amount_head(transformer_out) - 2),
            'time': self.time_head(transformer_out)
        }


def load_model(path: str, device: str = 'cpu'):
    full_path = f'saves/{path}'
    checkpoint = torch.load(full_path, map_location=device, weights_only=True)
    config = checkpoint['config']
    model = ModelLegacy(**config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return model


def main():
    device = "cuda"
    dataset = load_dataset("test_set/chunk_0.pkl")
    threshold = 0.5
    num_samples = 100
    seed = 0
    batch_size = 50

    lines = ["# Evaluation Results\n"]

    # Table 1: Concatenation vs Addition
    lines.append("| Model | Ps. Corr | Bin. Corr | Prec. | Rec. | Acc. |  F1  | Tok. Ratio | Seg Loss | Param Loss | Total Loss |")
    lines.append("|-------|----------|-----------|-------|------|------|------|------------|----------|------------|------------|")

    for model_name, label in [("concat_model.pt", "conc"), ("add_model.pt", "add")]:
        print(f"Evaluating {model_name}...")
        model = load_model(model_name, device=device)
        r = evaluate(model, dataset, num_samples, device, threshold, seed=seed, batch_size=batch_size)
        total_loss = r['segment_loss'] + r['param_loss']
        row = f"| {label}  |   {r['correlation']:.2f}   |  {r['binary_correlation']:>5.2f}    | {r['precision']:.2f}  | {r['recall']:.2f} | {r['accuracy']:.2f} | {r['f1']:.2f} |    {r['num_tokens_ratio']:.2f}    |   {r['segment_loss']:.2f}   |    {r['param_loss']:.2f}    |    {r['segment_loss'] + r['param_loss']:.2f}    |"
        lines.append(row)
        print(row)

    lines.append("\n")

    # Table 2: Different alpha values
    lines.append("| Alpha | Ps. Corr | Bin. Corr | Prec. | Rec. | Acc. |  F1  | Tok. Ratio | Seg Loss | Param Loss | Total Loss |")
    lines.append("|-------|----------|-----------|-------|------|------|------|------------|----------|------------|------------|")

    for alpha in [0, 0.25, 0.5, 0.75, 1]:
        model_name = f"alpha_{alpha}.pt"
        print(f"Evaluating {model_name}...")
        model = load_model(model_name, device=device)
        r = evaluate(model, dataset, num_samples, device, threshold, seed=seed, batch_size=batch_size)
        row = f"| {alpha:<5} | {r['correlation']:.2f}     | {r['binary_correlation']:>5.2f}      | {r['precision']:.2f}  | {r['recall']:.2f} | {r['accuracy']:.2f} | {r['f1']:.2f} | {r['num_tokens_ratio']:.2f}       | {r['segment_loss']:.2f}     | {r['param_loss']:.2f}       | {r['segment_loss'] + r['param_loss']:.2f}       |"
        lines.append(row)
        print(row)

    output_path = Path(__file__).parent.parent / "saves" / "model_evals_2.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
