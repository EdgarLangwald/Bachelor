import torch
from codebase.train import train_exhaustively
from codebase.model import Model
from codebase.utils import visualize_model, load_dataset

'''
if __name__ == '__main__':
    device = torch.device('cuda')
    train_exhaustively(
        batch_size=50,
        lr=1e-4,
        num_steps=50000,
        device="cuda",
        model=Model(360, 4, 4, 6, 1480, 0.1),
        print_every=100,
        model_path="400Mb.pt",
        dataset_path="400Mb",
        accumulation_steps=2,
        num_rotations=1,
        num_workers=0,
        alpha=0.7,
        add_checkpoints=7000,
        record_loss=100,
    )
'''


model = Model.load("400Mb.pt", "cuda")
dataset = load_dataset("dataset/chunk_0.pkl")

fig = visualize_model(
    model=model,
    dataset=dataset,
    num_plots=6,
    title="400Mb test set",
    device="cuda",
    exclude_context=False,
    show_notes=False,
    generate=True,
    ground_truth=False,
    seed=0,
    export_midi=True
    )

fig.show()
