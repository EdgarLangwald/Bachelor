import torch
import torch.nn.functional as F
import math
from typing import Dict, Tuple
from .data import SegmentToken
from .utils import TorchSegment

def compute_segment_loss(
    model_output: Dict[str, torch.Tensor],
    tokens: torch.Tensor,
    tgt_key_padding_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len = tokens.shape[0], tokens.shape[1]
    h = 1.0
    device = tokens.device

    pred_heights = model_output['height'][:, :-1, 0]
    pred_amounts = model_output['amount'][:, :-1, 0]
    pred_log_delta_times = torch.clamp(model_output['time'][:, :-1, 0], min=-10, max=10)
    pred_delta_times = torch.exp(pred_log_delta_times)
    pred_times = tokens[:, :-1, 2] + pred_delta_times

    target_times = tokens[:, 1:, 2]
    target_heights = tokens[:, 1:, 0]
    target_amounts = tokens[:, 1:, 1]

    prev_times = tokens[:, :-1, 2]
    prev_heights = tokens[:, :-1, 0]

    mask = ~tgt_key_padding_mask[:, 1:]

    # OPTIMIZATION 1: Vectorize Loop 1 - Find valid indices without Python loops
    valid_mask = mask[:, 1:]  # Exclude position 0 (positions 1 to seq_len-2)
    batch_indices, time_indices_relative = torch.where(valid_mask)
    time_indices = time_indices_relative + 1  # Adjust since we sliced from position 1

    if len(batch_indices) == 0:
        return torch.tensor(0.0, device=device)

    num_valid = len(batch_indices)

    x_starts = prev_times[batch_indices, time_indices]
    y_starts = prev_heights[batch_indices, time_indices]
    x_ends_pred = pred_times[batch_indices, time_indices]
    y_ends_pred = pred_heights[batch_indices, time_indices]
    amounts_pred = pred_amounts[batch_indices, time_indices]
    x_ends_gt = target_times[batch_indices, time_indices]
    y_ends_gt = target_heights[batch_indices, time_indices]
    amounts_gt = target_amounts[batch_indices, time_indices]

    x_min_ends = torch.min(x_ends_pred, x_ends_gt)
    x_max_ends = torch.max(x_ends_pred, x_ends_gt)

    L_vals = x_min_ends - x_starts

    # OPTIMIZATION 2: Divide each segment into even number of equal intervals
    # Number of intervals must be even for Simpson's rule
    num_intervals = 2 * torch.ceil(L_vals / h).long()
    num_intervals = torch.clamp(num_intervals, min=2)  # At least 2 intervals
    max_intervals = num_intervals.max().item()
    max_points = max_intervals + 1

    # Interval length (different per segment)
    delta = L_vals / num_intervals.float()  # Shape: (num_valid,)

    # Generate evaluation points with uniform spacing (delta per segment)
    pos_idx = torch.arange(max_points, device=device).unsqueeze(0)  # Shape: (1, max_points)
    eval_matrix = x_starts.unsqueeze(1) + pos_idx * delta.unsqueeze(1)  # Shape: (num_valid, max_points)

    # Mask points beyond the last valid point for each segment
    last_point_idx = num_intervals  # Number of points = num_intervals + 1, so last index = num_intervals
    valid_point_mask = pos_idx <= last_point_idx.unsqueeze(1)
    eval_matrix = torch.where(valid_point_mask, eval_matrix, torch.tensor(100.0, device=device))

    # Expand segment parameters for vectorized evaluation
    x_starts_exp = x_starts.unsqueeze(1).expand(-1, max_points)
    y_starts_exp = y_starts.unsqueeze(1).expand(-1, max_points)
    x_ends_pred_exp = x_ends_pred.unsqueeze(1).expand(-1, max_points)
    y_ends_pred_exp = y_ends_pred.unsqueeze(1).expand(-1, max_points)
    amounts_pred_exp = amounts_pred.unsqueeze(1).expand(-1, max_points)
    x_ends_gt_exp = x_ends_gt.unsqueeze(1).expand(-1, max_points)
    y_ends_gt_exp = y_ends_gt.unsqueeze(1).expand(-1, max_points)
    amounts_gt_exp = amounts_gt.unsqueeze(1).expand(-1, max_points)

    seg_pred = TorchSegment(x_starts_exp, y_starts_exp, x_ends_pred_exp, y_ends_pred_exp, amounts_pred_exp)
    seg_gt = TorchSegment(x_starts_exp, y_starts_exp, x_ends_gt_exp, y_ends_gt_exp, amounts_gt_exp)

    values_pred = seg_pred(eval_matrix)
    values_gt = seg_gt(eval_matrix)
    sq_diff = (values_pred - values_gt) ** 2

    # OPTIMIZATION 3: Apply Simpson's rule with vectorized weights
    # Simpson weights: [1, 4, 2, 4, 2, ..., 4, 1]
    weights = torch.where(pos_idx % 2 == 1, 4.0, 2.0)  # Odd positions get 4, even get 2
    weights = weights.expand(num_valid, -1).clone()
    weights[:, 0] = 1.0  # First point always 1

    # Set last valid point weight to 1
    seg_idx_range = torch.arange(num_valid, device=device)
    weights[seg_idx_range, last_point_idx] = 1.0

    # Apply mask to zero out invalid positions
    weights = weights * valid_point_mask.float()

    # Apply Simpson's rule: sum of (weight * sq_diff * delta / 6)
    loss_overlap = (weights * sq_diff * delta.unsqueeze(1) / 6).sum(dim=1)

    # Add mismatch penalty and normalize
    loss_mismatch = torch.abs(x_max_ends - x_min_ends)
    # loss_normalized = (loss_overlap + loss_mismatch) / (x_ends_gt - x_starts) # NORMALIZED
    loss_normalized = loss_overlap + loss_mismatch # UNNORMALIZED

    return loss_normalized.mean()


def compute_param_loss(
    model_output: Dict[str, torch.Tensor],
    tokens: torch.Tensor,
    tgt_key_padding_mask: torch.Tensor,
) -> torch.Tensor:

    epsilon = 1e-6

    pred_heights = model_output['height'][:, :-1, 0]
    pred_amounts = model_output['amount'][:, :-1, 0]
    pred_log_delta_times = torch.clamp(model_output['time'][:, :-1, 0], min=-10, max=10)

    target_times = tokens[:, 1:, 2]
    target_heights = tokens[:, 1:, 0]
    target_amounts = tokens[:, 1:, 1]

    prev_times = tokens[:, :-1, 2]
    target_delta_times = torch.clamp(target_times - prev_times, min=epsilon)
    target_log_delta_times = torch.log(target_delta_times)

    mask = ~tgt_key_padding_mask[:, 1:]

    height_loss = F.mse_loss(pred_heights[mask], target_heights[mask])
    amount_loss = F.mse_loss(pred_amounts[mask], target_amounts[mask])
    time_loss = F.mse_loss(pred_log_delta_times[mask], target_log_delta_times[mask])

    param_loss = 2.0 * height_loss + 1 * amount_loss + 0.5 * time_loss

    return param_loss


def step(model, batch, optimizer, device, alpha=0.5):
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

    param_loss = compute_param_loss(model_output, tokens, tgt_key_padding_mask)

    if alpha!=0:
        segment_loss = compute_segment_loss(model_output, tokens, tgt_key_padding_mask)
        total_loss = alpha * segment_loss + (1 - alpha) * param_loss
    else:
        segment_loss = torch.tensor(0.0, device=device)
        total_loss = param_loss

    loss_dict = {
        'total': total_loss.item(),
        'segment': segment_loss.item(),
        'param': param_loss.item()
    }

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss_dict


def train(batch_size, lr, num_steps, device, model=None, print_every=100, dataset=None, model_path=None, dataset_path=None, alpha=0.5):
    from .utils import load_dataset
    from torch.utils.data import DataLoader
    from .data import collate_fn
    from .model import Model
    import os

    if dataset is None and dataset_path is None:
        dataset_path = "dataset.pkl"
    if dataset is None:
        dataset = load_dataset(dataset_path) # type: ignore
        print(f"Using {dataset_path}")

    if model_path is None:
        model_path = "model.pt"
    if model is None:
        model = Model.load(model_path, device=device)
        print(f"Using {model_path}")
    else:
        print("Using provided model")

    assert dataset, "No dataset provided"
    assert model, "No model provided"

    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    print("Creating dataloader...")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    window_losses = {'total': 0.0, 'segment': 0.0, 'param': 0.0}
    step_count = 0

    print("Starting training loop...")
    while step_count < num_steps:
        for batch in dataloader:
            loss_dict = step(model, batch, optimizer, device, alpha=alpha)

            if any(math.isnan(v) for v in loss_dict.values()):
                print(f"NaN loss!", flush=True)
                continue

            for key in window_losses:
                window_losses[key] += loss_dict[key]
            step_count += 1

            if step_count % print_every == 0:
                print(f"Step {step_count}/{num_steps}")
                print(f"  Total: {window_losses['total']/print_every:.4f} | Segment: {window_losses['segment']/print_every:.4f} | Param: {window_losses['param']/print_every:.4f}")
                for key in window_losses:
                    window_losses[key] = 0.0

            if step_count >= num_steps:
                break

    print(f"Saving model to {model_path}")
    model.save(model_path)
    return


def train_exhaustively(
    batch_size,
    lr,
    num_steps,
    device,
    model=None,
    model_path="model.pt",
    dataset_path="dataset",
    print_every=100,
    accumulation_steps=4,
    num_rotations=1,
    alpha=0.5,
    num_workers=0,
    add_checkpoints=None,
    record_loss=None,
    start_training_at=0,
    ema_decay=0.99
):
    from .utils import load_dataset
    from torch.utils.data import DataLoader
    from .data import collate_fn
    from .model import Model
    from pathlib import Path
    import math
    import os
    import time

    min_lr = lr / 100
    weight_decay = 0.01
    grad_clip = 1.0
    num_workers = num_workers

    effective_batch_size = batch_size * accumulation_steps

    device = torch.device(device) if isinstance(device, str) else device

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if model is None:
        model = Model.load(model_path, device=device)
    else:
        model = model.to(device)

    dataset_full_path = Path("saves") / dataset_path
    chunk_paths = sorted(list(dataset_full_path.glob("chunk_*.pkl")))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk files found in {dataset_path}")
    print(f"Found {len(chunk_paths)} chunk files in {dataset_path}", flush=True)

    total_steps = num_steps * len(chunk_paths) * num_rotations
    warmup_steps = total_steps // 100

    optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr, weight_decay=weight_decay, eps=1e-5)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    def get_lr(step):
        if step < warmup_steps:
            return min_lr + (lr - min_lr) * step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    window_losses = {'total': 0.0, 'segment': 0.0, 'param': 0.0}
    chunks_done = start_training_at - 1 if start_training_at > 0 else 0
    step_count = chunks_done * num_steps
    accumulation_counter = 0
    optimizer.zero_grad()

    loss_history = [] if record_loss is not None else None

    segment_loss_ema = None
    param_loss_ema = None
    start_rotation = chunks_done // len(chunk_paths)
    start_chunk = chunks_done % len(chunk_paths)

    print(f"Training: warmup={warmup_steps}, effective_batch={batch_size}x{accumulation_steps}={effective_batch_size}, workers={num_workers}, rotations={num_rotations}", flush=True)
    print(f"Total steps: {num_steps} steps/chunk × {len(chunk_paths)} chunks × {num_rotations} rotations = {total_steps} steps", flush=True)
    if start_training_at > 0:
        print(f"Resuming at chunk {start_training_at} (rotation {start_rotation+1}, step {chunks_done * num_steps + 1})", flush=True)
    print(f"Optimizations: mixed_precision={'ON' if scaler else 'OFF'}, cudnn_benchmark=ON", flush=True)

    rotation = start_rotation

    while rotation < num_rotations:
        chunk_idx = start_chunk if rotation == start_rotation else 0

        while chunk_idx < len(chunk_paths):
            print(f"Loading chunk {chunk_idx+1}/{len(chunk_paths)}...", flush=True)
            chunk_dataset = load_dataset(str(chunk_paths[chunk_idx].relative_to(Path("saves"))))
            print(f"Creating dataloader for chunk {chunk_idx+1}...", flush=True)
            dataloader = DataLoader(
                chunk_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
            )

            print(f"Starting training on chunk {chunk_idx+1}...", flush=True)
            window_start = time.time()
            first_batch = True
            chunk_steps = 0
            batch_counter = 0
            while chunk_steps < num_steps:
                for batch in dataloader:
                    batch_counter += 1
                    if first_batch:
                        load_elapsed = time.time() - window_start
                        print(f"Data loaded, time: {int(load_elapsed)}s", flush=True)
                        window_start = time.time()
                        first_batch = False
                    model.train()

                    notes = batch['notes'].to(device)
                    tokens = batch['tokens'].to(device)
                    src_key_padding_mask = batch['src_key_padding_mask'].to(device)
                    tgt_key_padding_mask = batch['tgt_key_padding_mask'].to(device)

                    seq_len = tokens.size(1)
                    tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

                    if scaler:
                        with torch.amp.autocast('cuda'):
                            model_output = model(
                                notes, tokens,
                                src_key_padding_mask=src_key_padding_mask,
                                tgt_mask=tgt_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask
                            )

                            if alpha == 1:
                                segment_loss = compute_segment_loss(model_output, tokens, tgt_key_padding_mask)
                                param_loss = torch.tensor(0.0, device=device)
                                loss = segment_loss
                            elif alpha == 0:
                                segment_loss = torch.tensor(0.0, device=device)
                                param_loss = compute_param_loss(model_output, tokens, tgt_key_padding_mask)
                                loss = param_loss
                            else:
                                segment_loss = compute_segment_loss(model_output, tokens, tgt_key_padding_mask)
                                param_loss = compute_param_loss(model_output, tokens, tgt_key_padding_mask)
                                segment_loss_ema = segment_loss.item() if segment_loss_ema is None else ema_decay * segment_loss_ema + (1 - ema_decay) * segment_loss.item()
                                param_loss_ema = param_loss.item() if param_loss_ema is None else ema_decay * param_loss_ema + (1 - ema_decay) * param_loss.item()
                                norm_segment = segment_loss / (segment_loss_ema + 1e-8)
                                norm_param = param_loss / (param_loss_ema + 1e-8)
                                loss = alpha * norm_segment + (1 - alpha) * norm_param

                            loss_dict = {
                                'total': loss.item(),
                                'segment': segment_loss.item(),
                                'param': param_loss.item()
                            }

                        if any(math.isnan(v) for v in loss_dict.values()):
                            print(f"NaN loss! tokens={batch['tokens']}", flush=True)
                            optimizer.zero_grad()
                            continue

                        loss = loss / accumulation_steps
                        scaler.scale(loss).backward()
                    else:
                        model_output = model(
                            notes, tokens,
                            src_key_padding_mask=src_key_padding_mask,
                            tgt_mask=tgt_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask
                        )

                        if alpha == 1:
                            segment_loss = compute_segment_loss(model_output, tokens, tgt_key_padding_mask)
                            param_loss = torch.tensor(0.0, device=device)
                            loss = segment_loss
                        elif alpha == 0:
                            segment_loss = torch.tensor(0.0, device=device)
                            param_loss = compute_param_loss(model_output, tokens, tgt_key_padding_mask)
                            loss = param_loss
                        else:
                            segment_loss = compute_segment_loss(model_output, tokens, tgt_key_padding_mask)
                            param_loss = compute_param_loss(model_output, tokens, tgt_key_padding_mask)
                            segment_loss_ema = segment_loss.item() if segment_loss_ema is None else ema_decay * segment_loss_ema + (1 - ema_decay) * segment_loss.item()
                            param_loss_ema = param_loss.item() if param_loss_ema is None else ema_decay * param_loss_ema + (1 - ema_decay) * param_loss.item()
                            norm_segment = segment_loss / (segment_loss_ema + 1e-8)
                            norm_param = param_loss / (param_loss_ema + 1e-8)
                            loss = alpha * norm_segment + (1 - alpha) * norm_param

                        loss_dict = {
                            'total': loss.item(),
                            'segment': segment_loss.item(),
                            'param': param_loss.item()
                        }

                        if any(math.isnan(v) for v in loss_dict.values()):
                            print(f"NaN loss! tokens={batch['tokens']}", flush=True)
                            optimizer.zero_grad()
                            continue

                        loss = loss / accumulation_steps
                        loss.backward()

                    for key in window_losses:
                        window_losses[key] += loss_dict[key]
                    accumulation_counter += 1

                    if accumulation_counter % accumulation_steps == 0:
                        current_lr = get_lr(step_count)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr

                        if scaler:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                            optimizer.step()
                        optimizer.zero_grad()

                        step_count += 1
                        chunk_steps += 1

                        if add_checkpoints is not None and step_count % add_checkpoints == 0:
                            checkpoint_path = f"{model_path}_checkpoint_{step_count}.pt"
                            model.save(checkpoint_path)
                            print(f"Checkpoint saved to {checkpoint_path}", flush=True)

                        if record_loss is not None and step_count % record_loss == 0:
                            loss_snapshot = {key: val / (record_loss * accumulation_steps) for key, val in window_losses.items()}
                            loss_history.append({'step': step_count, 'losses': loss_snapshot, 'lr': current_lr})

                        if step_count % print_every == 0:
                            window_elapsed = time.time() - window_start
                            mins, secs = divmod(int(window_elapsed), 60)
                            print(f"Step {step_count}/{total_steps}, LR: {current_lr:.6f}", flush=True)
                            print(f"  Total: {window_losses['total']/(print_every*accumulation_steps):.4f} | Segment: {window_losses['segment']/(print_every*accumulation_steps):.4f} | Param: {window_losses['param']/(print_every*accumulation_steps):.4f} | time: {mins:02d}:{secs:02d}s", flush=True)
                            for key in window_losses:
                                window_losses[key] = 0.0
                            window_start = time.time()
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                        if chunk_steps >= num_steps:
                            break

                    if chunk_steps >= num_steps:
                        break

            del chunk_dataset
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            chunk_idx += 1

        rotation += 1

    model.save(model_path)
    print(f"Training complete! Model saved to {model_path}", flush=True)

    if record_loss is not None and loss_history:
        from .utils import save_pkl
        loss_path = f"{model_path}_loss_history.pkl"
        save_pkl(loss_history, loss_path)
        print(f"Loss history saved to saves/{loss_path}", flush=True)
