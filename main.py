import time
import torch
from codebase.train import compute_segment_loss, compute_param_loss
from codebase.model import Model
from codebase.utils import load_dataset, visualize_model, visualize_teacher_forcing
from codebase.data import collate_fn
from torch.utils.data import DataLoader

device = torch.device('cuda')
model = Model.load("test_model.pt", device=device)
dataset = load_dataset("complete_dataset_fixed/chunk_0.pkl")

fig = visualize_model(model, dataset, 12, "cuda", False, False, True)
fig.show()

fig = visualize_teacher_forcing