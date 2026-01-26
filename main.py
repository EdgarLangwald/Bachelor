import torch
from codebase.model import Model
from codebase.train import train_exhaustively
from codebase.preprocessing import create_dataset


if __name__ == '__main__':
    dataset = create_dataset(
        split="train",
        dataset_path="maestro-v3.0.0", 
        seg_fit_tightness=0.12,
        nocturnes=False,
        track_idx=None,
        num_workers=7,
        max_chunk_size_mb=10000,
        output_file="dataset"
    )
    
    device = torch.device('cuda')
    train_exhaustively(
        batch_size=100,
        lr=1e-4,
        num_steps=500000,
        device="cuda",
        model=Model(516, 6, 6, 8, 2064, 0.1),
        print_every=100,
        model_path="HPC_model.pt",
        dataset_path="dataset",
        accumulation_steps=1,
        num_rotations=1,
        num_workers=7,
        alpha=0.5,
        add_checkpoints=50000,
        record_loss=500,
    )
