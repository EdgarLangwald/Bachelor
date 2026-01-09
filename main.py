import torch
from codebase.train import compute_segment_loss
from codebase.model import Model
from codebase.utils import load_chunk_all

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading model...")
model = Model.load("model.pt", device=device)
model.eval()

print("Loading dataset...")
dataset = load_chunk_all("complete_dataset/chunk_0.pkl")

print("\nTesting optimized segment loss computation...\n")

notes, tokens = dataset[0]

notes_tensor = torch.tensor(
    [[note.start, note.duration, note.pitch, note.velocity] for note in notes],
    dtype=torch.float32
).unsqueeze(0).to(device)

tokens_tensor = torch.tensor(
    [[t.height, t.amount, t.time] for t in tokens],
    dtype=torch.float32
).unsqueeze(0).to(device)

seq_len = tokens_tensor.size(1)
tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
tgt_key_padding_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)

with torch.no_grad():
    model_output = model(
        notes_tensor, tokens_tensor,
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_key_padding_mask
    )

with torch.no_grad():
    loss = compute_segment_loss(model_output, tokens_tensor, tgt_key_padding_mask)
    print(f"Segment loss: {loss.item():.6f}")
    print("Optimized computation successful!")

print("\nTest complete!")
