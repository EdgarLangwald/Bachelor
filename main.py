import torch
from codebase.model import Model
from codebase.utils import load_pkl, visualize_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

dataset = load_pkl('nocturnes.pkl')
model = Model.load("saves/model.pt").to(device)

fig = visualize_model(model, dataset, num_plots=49, device=device, exclude_context=True)
fig.show()
