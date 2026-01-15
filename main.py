import time
import torch
from codebase.train import compute_segment_loss, compute_param_loss
from codebase.model import Model
from codebase.utils import load_dataset, visualize_model, visualize_teacher_forcing
from codebase.data import collate_fn
from torch.utils.data import DataLoader

device = torch.device('cuda')
model = Model.load("nocturnes_unnormalized.pt", device=device)
dataset = load_dataset("nocturnes/chunk_0.pkl")

'''
fig = visualize_model(model, dataset, 12, "cuda", False, False, True)
fig.show()
'''

fig = visualize_teacher_forcing(model, dataset, device, "Unnormalized model", seed=2)
fig.show()