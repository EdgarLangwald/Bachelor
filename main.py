import torch
from codebase.model import Model
from codebase.data import Dataset
from codebase.train import train, train_exhaustively
from codebase.inference import generate, tokens_to_segs
from codebase.utils import load_dataset, visualize_model, visualize_teacher_forcing, plot_loss, save_pkl
from codebase.preprocessing import create_dataset
from codebase.evaluate import evaluate
from codebase.miscellaneous.explore_curve_families import plot_track_segments

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    dataset = load_dataset("test_set/chunk_0.pkl")
    model = Model.load('HPC_model.pt', device=device)
    fig = visualize_model(
        model=model,
        dataset=dataset,
        num_plots=12,
        device=device,
        exclude_context=False,
        show_notes=False,
        generate=True,
        ground_truth=True,
        seed=0
        )
    fig.show()
