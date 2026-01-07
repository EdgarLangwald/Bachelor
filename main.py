import torch
from test_profiler import run_profiled_training

run_profiled_training(
    batch_size=8,
    num_steps=50,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    dataset_path='saves/dataset.pkl',
    model_path='saves/model.pt',
    num_workers=4,
    log_every=5
)
