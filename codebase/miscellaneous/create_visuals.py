import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize_scalar
import bisect
from typing import List, Tuple
from codebase.preprocessing import get_paths, get_track, segment
from codebase.data import PedalEvent


# Helper functions extracted/adapted from preprocessing.py

def perpendicular_distance(point, line_start, line_end):
    if np.array_equal(line_start, line_end):
        return np.linalg.norm(point - line_start)
    return np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)

def rdp(points, eps):
    dmax = 0
    index = 0
    end = len(points) - 1

    for i in range(1, end):
        d = perpendicular_distance(points[i], points[0], points[end])
        if d > dmax:
            index = i
            dmax = d

    if dmax > eps:
        left = rdp(points[:index+1], eps)
        right = rdp(points[index:], eps)
        return np.vstack((left[:-1], right))
    else:
        return np.array([points[0], points[end]])

def prepare_for_rdp(dense_time, dense_value, segment_time, segment_value):
    non_zero_mask = dense_value != 0
    dense_time_no_zeros = dense_time[non_zero_mask]
    dense_value_no_zeros = dense_value[non_zero_mask]

    modified_original_time = list(segment_time)
    modified_original_value = list(segment_value)

    for i in range(len(segment_value)):
        if segment_value[i] == 0:
            if i + 1 < len(segment_value):
                next_time = segment_time[i + 1]
                if next_time - segment_time[i] > 0.03:
                    modified_original_value[i + 1] = 0.0

    combined_time = np.concatenate([dense_time_no_zeros, modified_original_time])
    combined_value = np.concatenate([dense_value_no_zeros, modified_original_value])

    zero_times = set(segment_time[i] for i in range(len(segment_value)) if segment_value[i] == 0)

    keep_mask = np.ones(len(combined_time), dtype=bool)
    for i in range(len(combined_time)):
        for zero_time in zero_times:
            if abs(combined_time[i] - zero_time) < 0.03 and combined_value[i] != 0:
                keep_mask[i] = False
                break

    combined_time = combined_time[keep_mask]
    combined_value = combined_value[keep_mask]

    sort_indices = np.argsort(combined_time)
    combined_time = combined_time[sort_indices]
    combined_value = combined_value[sort_indices]

    return combined_time, combined_value

def improve_zero_values(xs, ys, segment_time, segment_value, threshold=0.03):
    non_zero_mask = ys != 0
    new_xs = list(xs[non_zero_mask])
    new_ys = list(ys[non_zero_mask])

    existing_times = set(new_xs)

    zero_points = []
    for i in range(len(segment_value)):
        if segment_value[i] == 0:
            if segment_time[i] not in existing_times:
                zero_points.append((segment_time[i], 0.0))
            if i + 1 < len(segment_value):
                next_time = segment_time[i + 1]
                if next_time - segment_time[i] > threshold and next_time not in existing_times:
                    zero_points.append((next_time, 0.0))

    all_points = list(zip(new_xs, new_ys)) + zero_points
    all_points.sort(key=lambda p: p[0])

    if not all_points:
        return np.array([]), np.array([])

    fixed_points = [all_points[0]]
    for i in range(1, len(all_points)):
        prev_time = fixed_points[-1][0]
        curr_time, curr_value = all_points[i]

        dt = curr_time - prev_time
        if dt < threshold:
            new_time = prev_time + threshold
            fixed_points.append((new_time, curr_value))
        else:
            fixed_points.append((curr_time, curr_value))

    new_xs = np.array([p[0] for p in fixed_points])
    new_ys = np.array([p[1] for p in fixed_points])

    return new_xs, new_ys


def get_pedal_at_time(pedal_events, t):
    """Get pedal value at time t (step function lookup)."""
    for i in range(len(pedal_events) - 1, -1, -1):
        if pedal_events[i].time <= t:
            return pedal_events[i].value
    return pedal_events[0].value

def filter_pedal_events(pedal_events, start_time, end_time):
    """Filter pedal events to window, adding boundary points."""
    filtered = [e for e in pedal_events if start_time < e.time < end_time]

    # Add start boundary
    start_value = get_pedal_at_time(pedal_events, start_time)
    filtered.insert(0, PedalEvent(start_time, start_value))

    # Add end boundary
    end_value = get_pedal_at_time(pedal_events, end_time)
    filtered.append(PedalEvent(end_time, end_value))

    time = np.array([e.time for e in filtered])
    value = np.array([e.value for e in filtered])
    return time, value


def get_segment_data(pedal_events: List[PedalEvent], start_time: float, end_time: float,
                     epsilon: float = 0.12, use_densification: bool = True, use_manual_adjustment: bool = True):
    """
    Extract intermediate data from the segmentation algorithm for a time window.
    Returns all intermediate steps for visualization.

    use_densification: If False, skip densification step (RDP runs on raw data only)
    use_manual_adjustment: If False, skip improve_zero_values step
    """
    # Filter pedal events to the window with boundary points
    time, value = filter_pedal_events(pedal_events, start_time, end_time)

    if len(time) < 2:
        return None

    # Step 1: Densification
    time_range = time[-1] - time[0]
    if time_range > 0:
        dense_time = np.linspace(time[0], time[-1], int(20 * time_range))
    else:
        dense_time = time

    idx = np.searchsorted(time, dense_time, side="right") - 1
    idx[idx < 0] = 0
    dense_value = value[idx]

    # Step 2: Prepare for RDP
    if use_densification:
        combined_time, combined_value = prepare_for_rdp(dense_time, dense_value, time, value)
    else:
        # Skip densification - use raw data directly
        combined_time, combined_value = time.copy(), value.copy()

    # Step 3: RDP
    points = np.column_stack((combined_time, combined_value))
    simplified = rdp(points, epsilon)
    rdp_xs, rdp_ys = simplified[:, 0], simplified[:, 1]

    # Step 4: Improve zero values (manual adjustment)
    if use_manual_adjustment:
        final_xs, final_ys = improve_zero_values(rdp_xs, rdp_ys, time, value)
    else:
        final_xs, final_ys = rdp_xs.copy(), rdp_ys.copy()

    return {
        'raw_time': time,
        'raw_value': value,
        'dense_time': dense_time,
        'dense_value': dense_value,
        'combined_time': combined_time,
        'combined_value': combined_value,
        'rdp_xs': rdp_xs,
        'rdp_ys': rdp_ys,
        'final_xs': final_xs,
        'final_ys': final_ys
    }


def fit_segments(final_xs, final_ys, pedal_events, p=2):
    """Fit segment curves between selected points."""
    events = sorted(pedal_events, key=lambda e: e.time)
    times = [e.time for e in events]

    def pedal_func(x):
        i = bisect.bisect_right(times, x) - 1
        if i < 0:
            return events[0].value
        return events[i].value

    def objective(amount, x_start, y_start, x_end, y_end):
        seg = segment(x_start, y_start, x_end, y_end, amount)

        idx_start = bisect.bisect_left(times, x_start)
        idx_end = bisect.bisect_right(times, x_end)
        events_in_range = [PedalEvent(x_start, pedal_func(x_start))]
        events_in_range.extend(events[idx_start: idx_end])
        events_in_range.append(PedalEvent(x_end, pedal_func(x_end)))

        total_error = 0
        for i, e in enumerate(events_in_range[:-1]):
            error = seg.Lp_distance_on_const(e.value, p, e.time, events_in_range[i+1].time)
            total_error += error

        return total_error ** (1/p)

    segments_list = []
    for i in range(len(final_xs) - 1):
        x_start, y_start = final_xs[i], final_ys[i]
        x_end, y_end = final_xs[i+1], final_ys[i+1]

        if np.abs(y_end - y_start) < 0.1:
            amount = 0.5
        else:
            result = minimize_scalar(
                lambda a: objective(a, x_start, y_start, x_end, y_end),
                bounds=(-0.5, 2.5),
                method='bounded',
                options={'maxiter': 50}
            )
            amount = result.x

        segments_list.append(segment(x_start, y_start, x_end, y_end, amount))

    return segments_list


def create_step_function_trace(time, value, name='Pedal', color='black', width=1.5, dash=None):
    """Create a step function trace for plotly."""
    step_time = []
    step_value = []

    for i in range(len(time)):
        step_time.append(time[i])
        step_value.append(value[i])
        if i < len(time) - 1:
            step_time.append(time[i+1])
            step_value.append(value[i])

    return go.Scatter(
        x=step_time, y=step_value, mode='lines',
        name=name, line=dict(color=color, width=width, dash=dash)
    )


def plot_raw_pedal(pedal_events, start_time, end_time, seed=42):
    """Plot 1: Raw pedal data."""
    np.random.seed(seed)

    time, value = filter_pedal_events(pedal_events, start_time, end_time)

    fig = go.Figure()
    fig.add_trace(create_step_function_trace(time, value, 'Raw Pedal', 'black'))

    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Pedal Value',
        xaxis=dict(range=[start_time, end_time]),
        yaxis=dict(range=[-0.05, 1.05]),
        template='plotly_white',
        width=900, height=400
    )
    return fig


def plot_with_threshold(pedal_events, start_time, end_time, threshold=0.5, seed=42):
    """Plot 2: Raw pedal with threshold line and colored regions."""
    np.random.seed(seed)

    time, value = filter_pedal_events(pedal_events, start_time, end_time)

    fig = go.Figure()

    # Compute intervals above/below threshold first
    top_y = 1.03
    bottom_y = -0.03
    region_height = 0.02

    on_intervals = []
    off_intervals = []

    i = 0
    while i < len(time) - 1:
        is_on = value[i] >= threshold
        interval_start = time[i]
        while i < len(time) - 1 and (value[i] >= threshold) == is_on:
            i += 1
        interval_end = time[i]
        if is_on:
            on_intervals.append((interval_start, interval_end))
        else:
            off_intervals.append((interval_start, interval_end))

    # Draw On intervals at top (cornflower blue)
    for x0, x1 in on_intervals:
        fig.add_shape(
            type='rect', x0=x0, x1=x1,
            y0=top_y - region_height/2, y1=top_y + region_height/2,
            fillcolor='cornflowerblue', line=dict(width=0)
        )
    # Draw Off intervals at bottom (black)
    for x0, x1 in off_intervals:
        fig.add_shape(
            type='rect', x0=x0, x1=x1,
            y0=bottom_y - region_height/2, y1=bottom_y + region_height/2,
            fillcolor='black', line=dict(width=0)
        )

    # Raw pedal (no legend entry)
    trace = create_step_function_trace(time, value, 'Raw Pedal', 'black')
    trace.showlegend = False
    fig.add_trace(trace)

    # Threshold line (no annotation, added to legend instead)
    fig.add_hline(y=threshold, line=dict(color='gray', dash='dash', width=2))

    # Legend traces for On/Off/Threshold
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='cornflowerblue', symbol='square'),
        name='On'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=10, color='black', symbol='square'),
        name='Off'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='gray', dash='dash', width=2),
        name=f'Threshold={threshold}'
    ))

    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Pedal Value',
        xaxis=dict(range=[start_time, end_time]),
        yaxis=dict(range=[-0.08, 1.08]),
        template='plotly_white',
        width=900, height=400
    )
    return fig


def plot_with_densification(pedal_events, start_time, end_time, seed=42):
    """Plot 3: Raw pedal with densification points."""
    np.random.seed(seed)

    data = get_segment_data(pedal_events, start_time, end_time)
    if data is None:
        return None

    fig = go.Figure()

    # Raw pedal
    fig.add_trace(create_step_function_trace(
        data['raw_time'], data['raw_value'], 'Raw Pedal', 'black'))

    # Densification points (black, not transparent, smaller)
    fig.add_trace(go.Scatter(
        x=data['dense_time'], y=data['dense_value'],
        mode='markers', name='Densification Points',
        marker=dict(color='black', size=3)
    ))

    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Pedal Value',
        xaxis=dict(range=[start_time, end_time]),
        yaxis=dict(range=[-0.05, 1.05]),
        template='plotly_white',
        width=900, height=400
    )
    return fig


def plot_with_rdp(pedal_events, start_time, end_time, epsilon=0.12, seed=42, use_densification=True):
    """Plot 4: Raw pedal + densification + RDP selected points (blue)."""
    np.random.seed(seed)

    data = get_segment_data(pedal_events, start_time, end_time, epsilon, use_densification)
    if data is None:
        return None

    fig = go.Figure()

    # Raw pedal
    fig.add_trace(create_step_function_trace(
        data['raw_time'], data['raw_value'], 'Raw Pedal', 'black'))

    # Densification points (black, not transparent, smaller) - only if using densification
    if use_densification:
        fig.add_trace(go.Scatter(
            x=data['dense_time'], y=data['dense_value'],
            mode='markers', name='Densification Points',
            marker=dict(color='black', size=3)
        ))

    # RDP selected points
    fig.add_trace(go.Scatter(
        x=data['rdp_xs'], y=data['rdp_ys'],
        mode='markers', name='RDP Selected Points',
        marker=dict(color='blue', size=10, symbol='circle')
    ))

    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Pedal Value',
        xaxis=dict(range=[start_time, end_time]),
        yaxis=dict(range=[-0.05, 1.05]),
        template='plotly_white',
        width=900, height=400
    )
    return fig


def plot_with_zero_adjustment(pedal_events, start_time, end_time, epsilon=0.12, seed=42, use_densification=True, use_manual_adjustment=True):
    """Plot 5: Same as before + points adjusted by improve_zero_values (red)."""
    np.random.seed(seed)

    data = get_segment_data(pedal_events, start_time, end_time, epsilon, use_densification, use_manual_adjustment)
    if data is None:
        return None

    fig = go.Figure()

    # Raw pedal
    fig.add_trace(create_step_function_trace(
        data['raw_time'], data['raw_value'], 'Raw Pedal', 'black'))

    # Densification points - only if using densification
    if use_densification:
        fig.add_trace(go.Scatter(
            x=data['dense_time'], y=data['dense_value'],
            mode='markers', name='Densification Points',
            marker=dict(color='black', size=3)
        ))

    # RDP selected points (blue)
    fig.add_trace(go.Scatter(
        x=data['rdp_xs'], y=data['rdp_ys'],
        mode='markers', name='RDP Selected Points',
        marker=dict(color='blue', size=10, symbol='circle')
    ))

    # Find points that were adjusted (in final but different from rdp, or new zeros)
    adjusted_xs = []
    adjusted_ys = []
    rdp_set = set(zip(data['rdp_xs'], data['rdp_ys']))
    for x, y in zip(data['final_xs'], data['final_ys']):
        if (x, y) not in rdp_set:
            adjusted_xs.append(x)
            adjusted_ys.append(y)

    if adjusted_xs:
        fig.add_trace(go.Scatter(
            x=adjusted_xs, y=adjusted_ys,
            mode='markers', name='Adjusted Points',
            marker=dict(color='red', size=10, symbol='circle')
        ))

    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Pedal Value',
        xaxis=dict(range=[start_time, end_time]),
        yaxis=dict(range=[-0.05, 1.05]),
        template='plotly_white',
        width=900, height=400
    )
    return fig


def plot_with_segments(pedal_events, start_time, end_time, epsilon=0.12, seed=42, use_densification=True, use_manual_adjustment=True):
    """Plot 6: Final visualization with all selected points and fitted segments."""
    np.random.seed(seed)

    data = get_segment_data(pedal_events, start_time, end_time, epsilon, use_densification, use_manual_adjustment)
    if data is None:
        return None

    # Filter events for segment fitting
    filtered_events = [e for e in pedal_events if start_time <= e.time <= end_time]
    segments_list = fit_segments(data['final_xs'], data['final_ys'], filtered_events)

    fig = go.Figure()

    # Raw pedal
    fig.add_trace(create_step_function_trace(
        data['raw_time'], data['raw_value'], 'Raw Pedal', 'lightgray', width=1))

    # Fitted segments
    for i, seg in enumerate(segments_list):
        seg_times = np.linspace(seg.x_start, seg.x_end, 50)
        seg_values = [seg(t) for t in seg_times]
        fig.add_trace(go.Scatter(
            x=seg_times, y=seg_values,
            mode='lines', name=f'Segment {i+1}' if i == 0 else None,
            line=dict(color='green', width=2.5),
            showlegend=(i == 0),
            legendgroup='segments'
        ))

    # All selected points (blue)
    fig.add_trace(go.Scatter(
        x=data['final_xs'], y=data['final_ys'],
        mode='markers', name='Selected Points',
        marker=dict(color='blue', size=10, symbol='circle')
    ))

    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Pedal Value',
        xaxis=dict(range=[start_time, end_time]),
        yaxis=dict(range=[-0.05, 1.05]),
        template='plotly_white',
        width=900, height=400
    )
    return fig


def plot_segmentation_steps(pedal_events, start_time, end_time, step=5,
                            threshold=0.5, epsilon=0.12, seed=42, use_densification=True, use_manual_adjustment=True, title=None):
    """
    Unified visualization function with step parameter.

    step=0: Raw pedal only
    step=1: Raw pedal + threshold line with colored regions
    step=2: Raw pedal + densification points
    step=3: Raw pedal + densification + RDP points (blue)
    step=4: Raw pedal + densification + RDP points + adjusted points (red)
    step=5: Final with all points (blue) + fitted segments

    use_densification: If False, skip densification (for steps 3-5) to show impact
    use_manual_adjustment: If False, skip improve_zero_values step (for steps 4-5)
    title: Optional title for the plot
    """
    np.random.seed(seed)

    if step == 0:
        fig = plot_raw_pedal(pedal_events, start_time, end_time, seed)
    elif step == 1:
        fig = plot_with_threshold(pedal_events, start_time, end_time, threshold, seed)
    elif step == 2:
        fig = plot_with_densification(pedal_events, start_time, end_time, seed)
    elif step == 3:
        fig = plot_with_rdp(pedal_events, start_time, end_time, epsilon, seed, use_densification)
    elif step == 4:
        fig = plot_with_zero_adjustment(pedal_events, start_time, end_time, epsilon, seed, use_densification, use_manual_adjustment)
    elif step == 5:
        fig = plot_with_segments(pedal_events, start_time, end_time, epsilon, seed, use_densification, use_manual_adjustment)
    else:
        raise ValueError(f"step must be 0-5, got {step}")

    if title is not None:
        fig.update_layout(title=title)
    return fig


def plot_full_track_scrollable(pedal_events, seed=42):
    """
    Plot the entire track with scroll functionality to scout for good sections.
    """
    np.random.seed(seed)

    time = np.array([e.time for e in pedal_events])
    value = np.array([e.value for e in pedal_events])

    fig = go.Figure()
    fig.add_trace(create_step_function_trace(time, value, 'Raw Pedal', 'black'))

    fig.update_layout(
        title='Full Track - Use Range Slider to Scout for Good Sections',
        xaxis_title='Time (s)',
        yaxis_title='Pedal Value',
        yaxis=dict(range=[-0.05, 1.05]),
        template='plotly_white',
        width=1200, height=500,
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='linear'
        )
    )
    return fig
