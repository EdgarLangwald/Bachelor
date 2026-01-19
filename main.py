import torch
from codebase.train import train_exhaustively
from codebase.model import Model
from codebase.utils import visualize_model, load_dataset


if __name__ == '__main__':
    device = torch.device('cuda')
    train_exhaustively(
        batch_size=50,
        lr=1e-4,
        num_steps=30000,
        device="cuda",
        model=Model(360, 4, 4, 6, 1480, 0.1),
        print_every=100,
        model_path="complete_30k.pt",
        dataset_path="complete_dataset",
        accumulation_steps=2,
        num_rotations=2,
        num_workers=0,
        alpha=0.7,
        add_checkpoints=7000,
        record_loss=100,
    )


'''
model = Model.load("complete_30k.pt", "cuda")
dataset = load_dataset("complete_dataset/chunk_45.pkl")

fig = visualize_model(
    model=model,
    dataset=dataset,
    num_plots=12,
    device="cuda",
    exclude_context=False,
    show_notes=False,
    generate=True,
    seed=None
    )

fig.show()
'''