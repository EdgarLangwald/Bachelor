import pickle
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List
from .data import SegmentEvent, SegmentToken
import math


def safe_to_pkl(data, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(filepath: str):
    load_path = Path("saves") / filepath
    with open(load_path, 'rb') as f:
        return pickle.load(f)


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


def visualize_model(model, dataset, num_plots: int, device: str = 'cpu', exclude_context: bool = False):
    from .inference import generate, tokens_to_segs
    from .data import Note

    rows = int(math.ceil(math.sqrt(num_plots)))
    cols = int(math.ceil(num_plots / rows))

    fig = make_subplots(
        rows=rows,
        cols=cols,
        vertical_spacing=0.02,
        horizontal_spacing=0.02
    )

    for i in range(num_plots):
        row = i // cols + 1
        col = i % cols + 1

        notes, tokens = dataset[0]

        if exclude_context:
            notes = [Note(start=0.0, duration=0.1, pitch=60, velocity=1)]
            tokens = []

        generated_tokens = generate(model, notes, max_length=127, device=device)
        generated_segs = tokens_to_segs(generated_tokens)
        ground_truth_segs = tokens_to_segs(tokens)

        gt_times, gt_values = segs_to_curve(ground_truth_segs)
        gen_times, gen_values = segs_to_curve(generated_segs)

        fig.add_trace(go.Scatter(
            x=gt_times,
            y=gt_values,
            mode='lines',
            name='Ground Truth',
            line=dict(color='blue', width=0.75),
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=gen_times,
            y=gen_values,
            mode='lines',
            name='Generated',
            line=dict(color='red', width=0.75),
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=col)

        token_times = [token.time for token in generated_tokens]
        token_heights = [token.height for token in generated_tokens]
        fig.add_trace(go.Scatter(
            x=token_times,
            y=token_heights,
            mode='markers',
            name='Tokens',
            marker=dict(size=1.3, color='orange'),
            showlegend=False,
            hoverinfo='skip'
        ), row=row, col=col)

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    height_per_plot = 100
    width_per_plot = 125
    fig.update_layout(
        height=height_per_plot * rows,
        width=width_per_plot * cols,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode=False
    )

    return fig
