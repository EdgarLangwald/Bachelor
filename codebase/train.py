import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from .data import SegmentToken
from .utils import TorchSegment


def _segment_loss_single(x_start, y_start, x_end_pred, y_end_pred, amount_pred, x_end_gt, y_end_gt, amount_gt):
    seg_pred = TorchSegment(x_start, y_start, x_end_pred, y_end_pred, amount_pred)
    seg_gt = TorchSegment(x_start, y_start, x_end_gt, y_end_gt, amount_gt)

    x_min_end = torch.min(x_end_pred, x_end_gt)
    x_max_end = torch.max(x_end_pred, x_end_gt)

    h = 1.0
    L = x_min_end - x_start
    n = (L / h).int()

    L_overlap = torch.tensor(0.0, device=x_start.device)

    if L < h:
        n = torch.tensor(0, device=x_start.device)

    for i in range(n.item()):
        a = x_start + i * h
        b = x_start + (i + 1) * h
        x0 = a
        x1 = (a + b) / 2
        x2 = b

        f0 = (seg_pred(x0) - seg_gt(x0))**2
        f1 = (seg_pred(x1) - seg_gt(x1))**2
        f2 = (seg_pred(x2) - seg_gt(x2))**2

        L_overlap = L_overlap + (b - a) / 6 * (f0 + 4*f1 + f2)

    remainder = L - h * n
    if remainder > 1e-6 or n == 0:
        a = x_start + n * h
        b = x_min_end
        x0 = a
        x1 = (a + b) / 2
        x2 = b

        f0 = (seg_pred(x0) - seg_gt(x0))**2
        f1 = (seg_pred(x1) - seg_gt(x1))**2
        f2 = (seg_pred(x2) - seg_gt(x2))**2

        L_overlap = L_overlap + (b - a) / 6 * (f0 + 4*f1 + f2)

    L_mismatch = torch.abs(x_max_end - x_min_end)
    L_normalized = (L_overlap + L_mismatch) / torch.clamp(x_end_gt - x_start, min=1e-6)

    return L_normalized


def compute_segment_loss(
    model_output: Dict[str, torch.Tensor],
    tokens: torch.Tensor,
    tgt_key_padding_mask: torch.Tensor,
) -> torch.Tensor:

    epsilon = 1e-6
    batch_size, seq_len = tokens.shape[0], tokens.shape[1]
    h = 1.0

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

    segment_loss = torch.tensor(0.0, device=tokens.device)
    num_valid_segments = 0

    for b in range(batch_size):
        for t in range(1, seq_len - 1):
            if not mask[b, t]:
                continue

            x_start = prev_times[b, t]
            y_start = prev_heights[b, t]
            x_end_pred = pred_times[b, t]
            y_end_pred = pred_heights[b, t]
            amount_pred = pred_amounts[b, t]
            x_end_gt = target_times[b, t]
            y_end_gt = target_heights[b, t]
            amount_gt = target_amounts[b, t]

            seg_loss = _segment_loss_single(
                x_start, y_start,
                x_end_pred, y_end_pred, amount_pred,
                x_end_gt, y_end_gt, amount_gt
            )

            segment_loss = segment_loss + seg_loss
            num_valid_segments += 1

    segment_loss = segment_loss / max(num_valid_segments, 1)

    return segment_loss


def compute_param_loss(
    model_output: Dict[str, torch.Tensor],
    tokens: torch.Tensor,
    tgt_key_padding_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:

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
    param_loss = 1.5 * param_loss

    loss_dict = {
        'param': param_loss.item(),
        'height': height_loss.item(),
        'amount': amount_loss.item(),
        'time': time_loss.item()
    }

    return param_loss, loss_dict


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

    segment_loss = compute_batch_segment_loss(model_output, tokens, tgt_key_padding_mask)
    param_loss, param_loss_dict = compute_param_loss(model_output, tokens, tgt_key_padding_mask)

    total_loss = alpha * segment_loss + (1 - alpha) * param_loss

    loss_dict = {
        'total': total_loss.item(),
        'segment': segment_loss.item(),
        **param_loss_dict
    }

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss_dict


def train(batch_size, lr, num_steps, device, model=None, print_every=100, dataset=None, model_path=None, dataset_path=None, alpha=0.5):
    from .utils import load_pkl
    from torch.utils.data import DataLoader
    from .data import collate_fn
    from .model import Model
    import os

    if dataset is None and dataset_path is None:
        dataset_path = "dataset.pkl"
    if dataset is None:
        dataset = load_pkl(dataset_path) # type: ignore
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

    window_losses = {'total': 0.0, 'segment': 0.0, 'param': 0.0, 'height': 0.0, 'amount': 0.0, 'time': 0.0}
    step_count = 0

    print("Starting training loop...")
    while step_count < num_steps:
        for batch in dataloader:

            loss_dict = step(model, batch, optimizer, device, alpha=alpha)
            for key in window_losses:
                window_losses[key] += loss_dict[key]
            step_count += 1

            if step_count % print_every == 0:
                print(f"Step {step_count}/{num_steps}")
                print(f"  Total: {window_losses['total']/print_every:.4f} | Segment: {window_losses['segment']/print_every:.4f} | Param: {window_losses['param']/print_every:.4f}")
                print(f"  Height: {window_losses['height']/print_every:.4f} | Amount: {window_losses['amount']/print_every:.4f} | Time: {window_losses['time']/print_every:.4f}")
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
    dataset=None,
    model_path="model.pt",
    dataset_path="dataset",
    print_every=100,
    accumulation_steps=4,
    num_rotations=1,
    alpha=0.5
):
    from .utils import load_pkl, load_chunk_all
    from torch.utils.data import DataLoader
    from .data import collate_fn
    from .model import Model
    from pathlib import Path
    import math
    import os

    min_lr = lr / 100
    weight_decay = 0.01
    grad_clip = 1.0
    num_workers = 0

    effective_batch_size = batch_size * accumulation_steps

    device = torch.device(device) if isinstance(device, str) else device

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    if model is None:
        model = Model.load(model_path, device=device)
    else:
        model = model.to(device)

    chunk_paths = None

    if dataset is None:
        dataset_full_path = Path("saves") / dataset_path
        if dataset_full_path.is_dir():
            chunk_paths = sorted(list(dataset_full_path.glob("chunk_*.pkl")))
            if not chunk_paths:
                raise FileNotFoundError(f"No chunk files found in {dataset_path}")
            print(f"Found {len(chunk_paths)} chunk files in {dataset_path}")
        else:
            dataset = load_pkl(dataset_path)
            print(f"Loaded single dataset from {dataset_path}")

    total_steps = num_steps * len(chunk_paths) * num_rotations if chunk_paths else num_steps
    warmup_steps = total_steps // 20

    optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    def get_lr(step):
        if step < warmup_steps:
            return min_lr + (lr - min_lr) * step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    window_losses = {'total': 0.0, 'segment': 0.0, 'param': 0.0, 'height': 0.0, 'amount': 0.0, 'time': 0.0}
    step_count = 0
    accumulation_counter = 0
    optimizer.zero_grad()

    print(f"Training: warmup={warmup_steps}, effective_batch={batch_size}x{accumulation_steps}={effective_batch_size}, workers={num_workers}, rotations={num_rotations}")
    print(f"Total steps: {num_steps} steps/chunk × {len(chunk_paths) if chunk_paths else 1} chunks × {num_rotations} rotations = {total_steps} steps")
    print(f"Optimizations: mixed_precision={'ON' if scaler else 'OFF'}, cudnn_benchmark=ON")

    if chunk_paths is not None:
        rotation = 0

        while rotation < num_rotations:
            chunk_idx = 0

            while chunk_idx < len(chunk_paths):
                print(f"Loading chunk {chunk_idx+1}/{len(chunk_paths)}...")
                chunk_dataset = load_chunk_all(str(chunk_paths[chunk_idx].relative_to(Path("saves"))))
                print(f"Creating dataloader for chunk {chunk_idx+1}...")
                dataloader = DataLoader(
                    chunk_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=num_workers,
                    pin_memory=True
                )

                print(f"Starting training on chunk {chunk_idx+1}...")
                chunk_steps = 0
                for batch in dataloader:
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
                            segment_loss = compute_batch_segment_loss(model_output, tokens, tgt_key_padding_mask)
                            param_loss, param_loss_dict = compute_param_loss(model_output, tokens, tgt_key_padding_mask)
                            loss = alpha * segment_loss + (1 - alpha) * param_loss
                            loss_dict = {
                                'total': loss.item(),
                                'segment': segment_loss.item(),
                                **param_loss_dict
                            }
                            loss = loss / accumulation_steps
                        scaler.scale(loss).backward()
                    else:
                        model_output = model(
                            notes, tokens,
                            src_key_padding_mask=src_key_padding_mask,
                            tgt_mask=tgt_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask
                        )
                        segment_loss = compute_batch_segment_loss(model_output, tokens, tgt_key_padding_mask)
                        param_loss, param_loss_dict = compute_param_loss(model_output, tokens, tgt_key_padding_mask)
                        loss = alpha * segment_loss + (1 - alpha) * param_loss
                        loss_dict = {
                            'total': loss.item(),
                            'segment': segment_loss.item(),
                            **param_loss_dict
                        }
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

                        if step_count % print_every == 0:
                            print(f"Step {step_count}/{total_steps}, LR: {current_lr:.6f}")
                            print(f"  Total: {window_losses['total']/(print_every*accumulation_steps):.4f} | Segment: {window_losses['segment']/(print_every*accumulation_steps):.4f} | Param: {window_losses['param']/(print_every*accumulation_steps):.4f}")
                            print(f"  Height: {window_losses['height']/(print_every*accumulation_steps):.4f} | Amount: {window_losses['amount']/(print_every*accumulation_steps):.4f} | Time: {window_losses['time']/(print_every*accumulation_steps):.4f}")
                            for key in window_losses:
                                window_losses[key] = 0.0

                        if chunk_steps >= num_steps:
                            break

                del chunk_dataset
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                chunk_idx += 1

            rotation += 1

    else:
        print("Creating dataloader...")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )

        print("Starting training loop...")
        while step_count < total_steps:
            for batch in dataloader:
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
                        segment_loss = compute_batch_segment_loss(model_output, tokens, tgt_key_padding_mask)
                        param_loss, param_loss_dict = compute_param_loss(model_output, tokens, tgt_key_padding_mask)
                        loss = alpha * segment_loss + (1 - alpha) * param_loss
                        loss_dict = {
                            'total': loss.item(),
                            'segment': segment_loss.item(),
                            **param_loss_dict
                        }
                        loss = loss / accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    model_output = model(
                        notes, tokens,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask
                    )
                    segment_loss = compute_batch_segment_loss(model_output, tokens, tgt_key_padding_mask)
                    param_loss, param_loss_dict = compute_param_loss(model_output, tokens, tgt_key_padding_mask)
                    loss = alpha * segment_loss + (1 - alpha) * param_loss
                    loss_dict = {
                        'total': loss.item(),
                        'segment': segment_loss.item(),
                        **param_loss_dict
                    }
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

                    if step_count % print_every == 0:
                        print(f"Step {step_count}/{total_steps}, LR: {current_lr:.6f}")
                        print(f"  Total: {window_losses['total']/(print_every*accumulation_steps):.4f} | Segment: {window_losses['segment']/(print_every*accumulation_steps):.4f} | Param: {window_losses['param']/(print_every*accumulation_steps):.4f}")
                        print(f"  Height: {window_losses['height']/(print_every*accumulation_steps):.4f} | Amount: {window_losses['amount']/(print_every*accumulation_steps):.4f} | Time: {window_losses['time']/(print_every*accumulation_steps):.4f}")
                        for key in window_losses:
                            window_losses[key] = 0.0

                    if step_count >= total_steps:
                        break

    model.save(model_path)
    print(f"Training complete! Model saved to {model_path}")