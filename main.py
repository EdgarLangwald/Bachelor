import torch
from codebase.model import Model
from codebase.data import Dataset
from codebase.train import train, train_exhaustively
from codebase.inference import generate, tokens_to_segs
from codebase.utils import load_pkl, plot_list, visualize_model, load_chunk_all, visualize_teacher_forcing
from codebase.preprocessing import create_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = load_chunk_all("complete_dataset/chunk_0.pkl")
model = Model.load('complete_model.pt', device=device)

fig = visualize_teacher_forcing(model, dataset)
fig.show()