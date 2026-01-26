import numpy as np
import plotly.graph_objects as go
from codebase.preprocessing import get_paths, get_track, pedals_to_segs, segment


def create_step_trace(time, value, name='Pedal', color='black', width=1.5):
    """Create a step function trace for plotly."""
    step_time = []
    step_value = []

    for i in range(len(time)):
        step_time.append(time[i])
        step_value.append(value[i])
        if i < len(time) - 1:
            step_time.append(time[i + 1])
            step_value.append(value[i])

    return go.Scatter(
        x=step_time, y=step_value, mode='lines',
        name=name, line=dict(color=color, width=width)
    )


def plot_track_segments(
    track_idx: int,
    show: list = [0, 100],
    split: str = "train",
    dataset_path: str = "maestro-v3.0.0",
    seg_fit_tightness: float = 0.12,
    nocturnes: bool = False,
):
    """
    Plot pedal step function with fitted segment curves and RDP points.

    Args:
        track_idx: Index of track to plot
        show: [start, end] time range to display
        split: "train", "validation", or "test"
        dataset_path: Path to MAESTRO dataset
        seg_fit_tightness: Epsilon for RDP algorithm
        nocturnes: Use nocturne subset
    """
    t_start, t_end = show

    paths = get_paths(split, dataset_path, nocturnes)
    path = paths[track_idx]
    print(f"Loading: {path}")

    pedal_events, note_events = get_track(path)
    segment_events = pedals_to_segs(pedal_events, epsilon=seg_fit_tightness)

    # Filter to show range
    segments_list = []
    for seg_ev in segment_events:
        if seg_ev.x_end >= t_start and seg_ev.x_start <= t_end:
            seg = segment(seg_ev.x_start, seg_ev.y_start, seg_ev.x_end, seg_ev.y_end, seg_ev.amount)
            segments_list.append(seg)

    time = np.array([e.time for e in pedal_events])
    value = np.array([e.value for e in pedal_events])
    mask = (time >= t_start) & (time <= t_end)
    time, value = time[mask], value[mask]

    fig = go.Figure()

    # Raw pedal step function (light gray)
    fig.add_trace(create_step_trace(time, value, 'Raw Pedal', 'lightgray', width=1))

    # Fitted segments (green)
    first_seg = True
    for seg in segments_list:
        seg_times = np.linspace(seg.x_start, seg.x_end, 50)
        seg_values = [seg(t) for t in seg_times]

        fig.add_trace(go.Scatter(
            x=seg_times, y=seg_values,
            mode='lines', name='Segment' if first_seg else None,
            line=dict(color='green', width=2.5),
            showlegend=first_seg,
            legendgroup='segments'
        ))
        first_seg = False

    # RDP selected points (blue)
    point_times = []
    point_heights = []
    for seg in segments_list:
        if not point_times:
            point_times.append(seg.x_start)
            point_heights.append(seg.y_start)
        point_times.append(seg.x_end)
        point_heights.append(seg.y_end)

    fig.add_trace(go.Scatter(
        x=point_times, y=point_heights,
        mode='markers', name='Selected Points',
        marker=dict(color='blue', size=10, symbol='circle')
    ))

    fig.update_layout(
        title=f'Track {track_idx}: {path.split("/")[-1]}',
        xaxis_title='Time (s)',
        yaxis_title='Pedal Value',
        xaxis=dict(range=[t_start, t_end], rangeslider=dict(visible=True)),
        yaxis=dict(range=[-0.05, 1.05]),
        template='plotly_white',
        width=1000, height=500
    )

    return fig


if __name__ == "__main__":
    fig = plot_track_segments(track_idx=0)
    fig.show()
