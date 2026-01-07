import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from .data import SegmentToken


def compute_loss(
    model_output: Dict[str, torch.Tensor],
    tokens: torch.Tensor,
    tgt_key_padding_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    epsilon = 1e-6

    pred_heights = model_output['height'][:, :-1, 0]
    pred_amounts = model_output['amount'][:, :-1, 0]
    pred_log_delta_times = model_output['time'][:, :-1, 0]

    target_heights = tokens[:, 1:, 0]
    target_amounts = tokens[:, 1:, 1]
    target_times = tokens[:, 1:, 2]
    prev_times = tokens[:, :-1, 2]
    target_delta_times = torch.clamp(target_times - prev_times, min=epsilon)
    target_log_delta_times = torch.log(target_delta_times)

    mask = ~tgt_key_padding_mask[:, 1:]

    height_loss = F.mse_loss(pred_heights[mask], target_heights[mask])
    amount_loss = F.mse_loss(pred_amounts[mask], target_amounts[mask])
    time_loss = F.mse_loss(pred_log_delta_times[mask], target_log_delta_times[mask])

    total_loss = 1 * height_loss + 0.2 * amount_loss + 1 * time_loss

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


def train(batch_size, lr, num_steps, device, model=None, print_every=100, dataset=None, model_path=None, dataset_path=None):
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

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True
    )
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
    accumulation_steps=4
):
    from .utils import load_pkl, load_chunk_all
    from torch.utils.data import DataLoader
    from .data import collate_fn
    from .model import Model
    from pathlib import Path
    import math
    import os

    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    min_lr = lr / 100
    weight_decay = 0.01
    grad_clip = 1.0
    num_workers = 6

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

    total_steps = num_steps * len(chunk_paths) if chunk_paths else num_steps
    warmup_steps = total_steps // 20

    optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    def get_lr(step):
        if step < warmup_steps:
            return min_lr + (lr - min_lr) * step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    window_loss = 0.0
    step_count = 0
    accumulation_counter = 0
    optimizer.zero_grad()

    print(f"Training: warmup={warmup_steps}, effective_batch={batch_size}x{accumulation_steps}={effective_batch_size}, workers={num_workers}")
    print(f"Optimizations: mixed_precision={'ON' if scaler else 'OFF'}, cudnn_benchmark=ON")

    if chunk_paths is not None:
        chunk_idx = 0

        while chunk_idx < len(chunk_paths):
            chunk_dataset = load_chunk_all(str(chunk_paths[chunk_idx].relative_to(Path("saves"))))
            dataloader = DataLoader(
                chunk_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )

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
                        loss, loss_dict = compute_loss(model_output, tokens, tgt_key_padding_mask)
                        loss = loss / accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    model_output = model(
                        notes, tokens,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask
                    )
                    loss, loss_dict = compute_loss(model_output, tokens, tgt_key_padding_mask)
                    loss = loss / accumulation_steps
                    loss.backward()

                window_loss += loss_dict['total']
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
                        avg_window_loss = window_loss / (print_every * accumulation_steps)
                        print(f"Step {step_count}/{total_steps}, Loss: {avg_window_loss:.4f}")
                        window_loss = 0.0

                    if chunk_steps >= num_steps:
                        break

            del chunk_dataset
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            chunk_idx += 1

    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )

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
                        loss, loss_dict = compute_loss(model_output, tokens, tgt_key_padding_mask)
                        loss = loss / accumulation_steps
                    scaler.scale(loss).backward()
                else:
                    model_output = model(
                        notes, tokens,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask
                    )
                    loss, loss_dict = compute_loss(model_output, tokens, tgt_key_padding_mask)
                    loss = loss / accumulation_steps
                    loss.backward()

                window_loss += loss_dict['total']
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
                        avg_window_loss = window_loss / (print_every * accumulation_steps)
                        print(f"Step {step_count}/{total_steps}, Loss: {avg_window_loss:.4f}")
                        window_loss = 0.0

                    if step_count >= total_steps:
                        break

    model.save(model_path)
    print(f"Training complete! Model saved to {model_path}")