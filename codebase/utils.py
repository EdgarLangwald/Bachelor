import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List
from .data import SegmentEvent, SegmentToken, PedalEvent, Note
import math
import torch


def save_pkl(data, filepath: str):
    save_path = Path("saves") / filepath
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def plot_loss(filepath: str, detailed: bool = True, plot_lr: bool = False, title: str = 'Training Loss', width: int = 800, height: int = 500):
    load_path = Path("saves") / filepath

    with open(load_path, 'rb') as f:
        loss_history = pickle.load(f)

    valid_entries = [e for e in loss_history if e['losses']['total'] is not None]
    steps = [entry['step'] for entry in valid_entries]
    total_losses = [entry['losses']['total'] for entry in valid_entries]

    fig = go.Figure()

    if detailed:
        segment_losses = [entry['losses']['segment'] for entry in valid_entries]
        param_losses = [entry['losses']['param'] for entry in valid_entries]
        sum_losses = [s + p for s, p in zip(segment_losses, param_losses)]

        fig.add_trace(go.Scatter(
            x=steps,
            y=sum_losses,
            mode='lines',
            name='Sum',
            line=dict(color='blue', width=3.4)
        ))

        fig.add_trace(go.Scatter(
            x=steps,
            y=segment_losses,
            mode='lines',
            name='Segment Loss',
            line=dict(color='red', width=3.4)
        ))

        fig.add_trace(go.Scatter(
            x=steps,
            y=param_losses,
            mode='lines',
            name='Param Loss',
            line=dict(color='green', width=3.4)
        ))
    else:
        fig.add_trace(go.Scatter(
            x=steps,
            y=total_losses,
            mode='lines',
            name='Total Loss',
            line=dict(color='blue', width=3.4)
        ))

    if plot_lr:
        lr_steps = [entry['step'] for entry in loss_history]
        lrs = [entry['lr'] for entry in loss_history]
        max_lr = max(lrs)
        lrs_normalized = [lr / max_lr for lr in lrs]
        fig.add_trace(go.Scatter(
            x=lr_steps,
            y=lrs_normalized,
            mode='lines',
            name=f'LR (normalized)',
            line=dict(color='purple', width=3.4, dash='7px 3px')
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=27)),
        xaxis_title='Steps',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, font=dict(size=20)),
        font=dict(size=20),
        width=width,
        height=height
    )
    if detailed:
        first_loss = segment_losses[0] + param_losses[0]
    else:
        first_loss = total_losses[0]
    fig.update_yaxes(
        title_text="Loss",
        range=[0, first_loss + 0.3]
    )

    return fig


def load_dataset(filepath: str):
    from .data import Dataset

    load_path = Path("saves") / filepath
    dataset = Dataset()

    with open(load_path, 'rb') as f:
        while True:
            try:
                notes, tokens = pickle.load(f)
                dataset.add_tracks(notes, tokens)
            except EOFError:
                break

    return dataset




def segs_to_curve(segments: List[SegmentEvent], num_points: int = 1000):
    from .preprocessing import segment

    if not segments:
        return np.array([]), np.array([])

    min_time = segments[0].x_start
    max_time = segments[-1].x_end

    times = np.linspace(min_time, max_time, num_points)
    values = np.zeros(num_points)

    for seg_event in segments:
        seg = segment(seg_event.x_start, seg_event.y_start, seg_event.x_end, seg_event.y_end, seg_event.amount)

        mask = (times >= seg.x_start) & (times <= seg.x_end)
        values[mask] = [seg(t) for t in times[mask]]

    return times, values


def segs_to_pedal(segments: List[SegmentEvent], sample_rate: int = 100) -> List[PedalEvent]:
    from .preprocessing import segment

    if not segments:
        return []

    min_time = segments[0].x_start
    max_time = segments[-1].x_end
    num_points = int((max_time - min_time) * sample_rate)

    times = np.linspace(min_time, max_time, num_points)
    pedal_events = []

    for t in times:
        for seg_event in segments:
            if seg_event.x_start <= t <= seg_event.x_end:
                seg = segment(seg_event.x_start, seg_event.y_start, seg_event.x_end, seg_event.y_end, seg_event.amount)
                pedal_events.append(PedalEvent(time=t, value=seg(t)))
                break

    return pedal_events


def export_to_midi(notes: List[Note], pedal_events: List[PedalEvent], filepath: str):
    import pretty_midi

    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    for note in notes:
        midi_note = pretty_midi.Note(
            velocity=max(1, note.velocity * 8),
            pitch=note.pitch + 20,
            start=note.start,
            end=note.start + note.duration
        )
        piano.notes.append(midi_note)

    for pedal in pedal_events:
        cc = pretty_midi.ControlChange(number=64, value=int(pedal.value * 127), time=pedal.time)
        piano.control_changes.append(cc)

    midi.instruments.append(piano)
    midi.write(filepath)


def plot_list(
    ground_truth_segs: List[SegmentEvent],
    generated_segs: List[SegmentEvent],
    title: str = "Pedal Comparison",
    tokens: List[SegmentToken] = None
):
    fig = go.Figure()

    gt_times, gt_values = segs_to_curve(ground_truth_segs)
    gen_times, gen_values = segs_to_curve(generated_segs)

    fig.add_trace(go.Scatter(
        x=gt_times,
        y=gt_values,
        mode='lines',
        name='Ground Truth',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=gen_times,
        y=gen_values,
        mode='lines',
        name='Generated',
        line=dict(color='red', width=2,)
    ))

    tokens = tokens[1:]
    if tokens is not None:
        token_times = [token.time for token in tokens]
        token_heights = [token.height for token in tokens]
        fig.add_trace(go.Scatter(
            x=token_times,
            y=token_heights,
            mode='markers',
            name='Tokens',
            marker=dict(size=8, color='orange')
        ))

    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Pedal Value",
        hovermode='x unified'
    )

    return fig


def visualize_teacher_forcing(model, dataset, device=None, title: str = "Teacher Forcing Visualization", seed=None):
    from .inference import tokens_to_segs
    from .data import SegmentToken
    from .preprocessing import segment
    import torch
    import random

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    notes, tokens = dataset()

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device) if isinstance(device, str) else device

    model.eval()
    notes_tensor = torch.tensor(
        [[[note.start, note.duration, note.pitch, note.velocity] for note in notes]],
        dtype=torch.float32
    ).to(device)
    tokens_tensor = torch.tensor(
        [[[t.height, t.amount, t.time] for t in tokens]],
        dtype=torch.float32
    ).to(device)

    tgt_len = len(tokens)
    tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1).bool()

    with torch.no_grad():
        model_output = model(notes_tensor, tokens_tensor, tgt_mask=tgt_mask)

    ground_truth_segs = tokens_to_segs(tokens)
    gt_times, gt_values = segs_to_curve(ground_truth_segs)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=gt_times,
        y=gt_values,
        mode='lines',
        name='Ground Truth',
        line=dict(color='lightblue', width=2)
    ))

    for i in range(1, tgt_len - 1):
        height = model_output['height'][0, i, 0].item()
        amount = model_output['amount'][0, i, 0].item()
        log_time_delta = model_output['time'][0, i, 0]
        time_delta = torch.exp(log_time_delta).item()
        pred_time = tokens[i].time + time_delta

        gt_token = tokens[i]

        fig.add_trace(go.Scatter(
            x=[gt_token.time],
            y=[gt_token.height],
            mode='markers',
            name='GT Token',
            marker=dict(size=8, color='blue'),
            showlegend=(i == 1),
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=[pred_time],
            y=[height],
            mode='markers',
            name='Generated Token',
            marker=dict(size=8, color='yellow'),
            showlegend=(i == 1),
            hoverinfo='skip'
        ))

        seg = segment(gt_token.time, gt_token.height, pred_time, height, amount)
        seg_times = np.linspace(seg.x_start, seg.x_end, 100)
        seg_values = [seg(t) for t in seg_times]

        fig.add_trace(go.Scatter(
            x=seg_times,
            y=seg_values,
            mode='lines',
            name='Predicted Segment',
            line=dict(color='red', width=1.5),
            showlegend=(i == 1),
            hoverinfo='skip'
        ))

    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Pedal Value",
        hovermode='x unified'
    )

    return fig


def visualize_model(model, dataset, num_plots: int, device: str = 'cpu', exclude_context: bool = False, show_notes: bool = True, generate: bool = True, seed=None, title: str = None, ground_truth: bool = True, export_midi: bool = False):
    from .inference import generate as generate_fn, tokens_to_segs
    import random

    assert not export_midi or (generate != ground_truth), "export_midi requires exactly one of generate or ground_truth to be True"
    assert not export_midi or title, "export_midi requires a title for the filename"

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    max_cols = 7
    if num_plots >= 35:
        cols = max_cols
    else:
        cols = min(max_cols, int(math.ceil(math.sqrt(num_plots))))

    rows = int(math.ceil(num_plots / cols))
    scale_factor = min(max_cols / cols, 3.5) * 1.5

    fig = make_subplots(
        rows=rows,
        cols=cols,
        vertical_spacing=0.015,
        horizontal_spacing=0.015
    )

    all_shapes = []

    for i in range(num_plots):
        row = i // cols + 1
        col = i % cols + 1

        notes, tokens = dataset()

        if exclude_context:
            notes = [Note(start=(10/88)*i, duration=10/88, pitch=i, velocity=1) for i in range(88)]
            tokens = []

        if ground_truth:
            ground_truth_segs = tokens_to_segs(tokens)
            gt_times, gt_values = segs_to_curve(ground_truth_segs)

        if generate:
            generated_tokens = generate_fn(model, notes, max_length=127, device=device)
            generated_segs = tokens_to_segs(generated_tokens)
            gen_times, gen_values = segs_to_curve(generated_segs)

        if export_midi:
            segs = generated_segs if generate else ground_truth_segs
            pedal_events = segs_to_pedal(segs)
            export_to_midi(notes, pedal_events, f"{title}_{i}.mid")

        if show_notes:
            axis_idx = (row - 1) * cols + col
            xref = 'x' if axis_idx == 1 else f'x{axis_idx}'
            yref = 'y' if axis_idx == 1 else f'y{axis_idx}'
            for note in notes:
                all_shapes.append(dict(
                    type='rect',
                    x0=note.start,
                    x1=note.start + note.duration,
                    y0=note.pitch / 88,
                    y1=(note.pitch + 1) / 88,
                    fillcolor='rgba(144, 238, 144, 0.85)',
                    line=dict(width=0),
                    xref=xref,
                    yref=yref
                ))

        if ground_truth:
            fig.add_trace(go.Scatter(
                x=gt_times,
                y=gt_values,
                mode='lines',
                name='Ground Truth',
                line=dict(color='blue', width=0.75 * scale_factor * (1 - 0.4 * show_notes)),
                showlegend=False,
                hoverinfo='skip'
            ), row=row, col=col)

        if generate:
            fig.add_trace(go.Scatter(
                x=gen_times,
                y=gen_values,
                mode='lines',
                name='Generated',
                line=dict(color='red', width=0.75 * scale_factor * (1 - 0.4 * show_notes)),
                showlegend=False,
                hoverinfo='skip'
            ), row=row, col=col)

            token_times = [token.time for token in generated_tokens[1:]]
            token_heights = [token.height for token in generated_tokens[1:]]
            fig.add_trace(go.Scatter(
                x=token_times,
                y=token_heights,
                mode='markers',
                name='Tokens',
                marker=dict(size=1.6 * scale_factor * (1 - 0.4 * show_notes), color='orange'),
                showlegend=False,
                hoverinfo='skip'
            ), row=row, col=col)

    if all_shapes:
        fig.update_layout(shapes=all_shapes)

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    height_per_plot = 70 * scale_factor
    width_per_plot = 125 * scale_factor
    fig.update_layout(
        title=title,
        height=height_per_plot * rows,
        width=width_per_plot * cols,
        showlegend=False,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        hovermode=False
    )

    return fig


class TorchSegment:
    def __init__(self, x_start, y_start, x_end, y_end, amount):
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.amount = torch.clamp(amount, 0, 2) * 3 - 0.5

    def a(self, x):
        return torch.clamp(x**3, min=0.1, max=10)

    def S1(self, x):
        return (x + 1e-8)**self.a(self.amount)

    def S2(self, x):
        return 1 - (1-x + 1e-8)**self.a(2-self.amount)

    def Single(self, x):
        return torch.where(self.amount > 1, self.S1(x), self.S2(x))

    def norm(self, x):
        return torch.clamp((x - self.x_start) / torch.clamp(self.x_end - self.x_start, min=1e-6), 0, 1)

    def scale(self, x):
        return self.y_start + x * (self.y_end - self.y_start)

    def __call__(self, x):
        mask = (x != 100.0)
        result = torch.where(mask, self.scale(self.Single(self.norm(x))), torch.tensor(0.0, device=x.device))
        return result
