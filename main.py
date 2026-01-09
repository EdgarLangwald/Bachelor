import torch
from codebase.model import Model
from codebase.train import train
from codebase.utils import load_chunk_all

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    dataset = load_chunk_all("complete_dataset/chunk_0.pkl")
    print(f"Dataset loaded: {len(dataset)} tracks")

    model = Model(300, 4, 3, 3, 1200, 0.1).to(device)
    print("Model created")

    train(
        batch_size=96,
        lr=0.8e-4,
        num_steps=200,
        device=device,
        model=model,
        print_every=1,
        dataset=dataset,
        model_path="test_model.pt",
        alpha=0.2
    )
