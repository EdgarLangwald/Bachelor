import torch
import torch.nn.functional as F
from torch.distributions import Beta, LogNormal
from typing import Dict, List, Tuple
from .data import SegmentToken


def compute_loss(
    model_output: Dict[str, torch.Tensor],
    tokens: torch.Tensor,
    tgt_key_padding_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    epsilon = 1e-6

    height_params = F.softplus(model_output['height'][:, :-1]) + epsilon
    amount_params = F.softplus(model_output['amount'][:, :-1]) + epsilon

    pred_log_delta_times = model_output['time'][:, :-1, 0]

    target_heights = tokens[:, 1:, 0]
    target_amounts = tokens[:, 1:, 1]
    target_times = tokens[:, 1:, 2]
    prev_times = tokens[:, :-1, 2]
    target_delta_times = torch.clamp(target_times - prev_times, min=epsilon)
    target_log_delta_times = torch.log(target_delta_times)

    mask = ~tgt_key_padding_mask[:, 1:]

    heights = torch.clamp(target_heights[mask], epsilon, 1.0 - epsilon)
    masked_height_params = height_params[mask]
    alpha_h, beta_h = masked_height_params[:, 0], masked_height_params[:, 1]
    height_loss = -Beta(alpha_h, beta_h).log_prob(heights).mean()

    amounts = torch.clamp(target_amounts[mask], epsilon, 1.0 - epsilon)
    masked_amount_params = amount_params[mask]
    alpha_a, beta_a = masked_amount_params[:, 0], masked_amount_params[:, 1]
    amount_loss = -Beta(alpha_a, beta_a).log_prob(amounts).mean()

    masked_pred_log_dt = pred_log_delta_times[mask]
    masked_target_log_dt = target_log_delta_times[mask]
    time_loss = F.mse_loss(masked_pred_log_dt, masked_target_log_dt)

    total_loss = 1 * height_loss + 0 * amount_loss + 1 * time_loss

    loss_dict = {
        'total': total_loss.item(),
        'height': height_loss.item(),
        'amount': amount_loss.item(),
        'time': time_loss.item()
    }

    return total_loss, loss_dict


def step(model, batch, optimizer, device):
    model.train()
    optimizer.zero_grad()

    notes = batch['notes'].to(device)
    tokens = batch['tokens'].to(device)
    src_key_padding_mask = batch['src_key_padding_mask'].to(device)
    tgt_key_padding_mask = batch['tgt_key_padding_mask'].to(device)

    seq_len = tokens.size(1)
    tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    model_output = model(
        notes, tokens,
        src_key_padding_mask=src_key_padding_mask,
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_key_padding_mask
    )

    loss, loss_dict = compute_loss(model_output, tokens, tgt_key_padding_mask)

    loss.backward()
    optimizer.step()

    return loss_dict['total']


def train(dataset_path, batch_size, lr, num_steps, device, model=None, print_every=100, dataset=None):
    from .utils import load_pkl
    from torch.utils.data import DataLoader
    from .data import collate_fn
    from .model import Model
    import os

    if dataset is None:
        dataset = load_pkl(dataset_path)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model_path = 'saves/model.pt'
    
    if model is not None:
        print(f"Using provided model")
    elif os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = Model.load(model_path, device=device)
    else:
        raise ValueError("No model provided and no saved model found")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    window_loss = 0.0
    step_count = 0

    while step_count < num_steps:
        for batch in dataloader:

            loss = step(model, batch, optimizer, device)
            window_loss += loss
            step_count += 1

            if step_count % print_every == 0:
                avg_window_loss = window_loss / print_every
                print(f"Step {step_count}/{num_steps}, Avg Loss: {avg_window_loss:.4f}")
                window_loss = 0.0

    print(f"Saving model to {model_path}")
    model.save(model_path)
    return