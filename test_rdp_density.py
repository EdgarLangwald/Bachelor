import numpy as np
import plotly.graph_objects as go
from codebase.preprocessing import get_track, pedals_to_segs
from codebase.utils import segs_to_curve
import sys

def pedals_to_segs_with_density(pedal_events, epsilon=0.12, density=100):
    import bisect
    from codebase.data import PedalEvent, SegmentEvent
    from codebase.preprocessing import segment
    from scipy.optimize import minimize_scalar

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

    def improve_zero_values(xs, ys, segment_time, segment_value):
        non_zero_mask = ys != 0
        new_xs = list(xs[non_zero_mask])
        new_ys = list(ys[non_zero_mask])

        zero_points = []
        for i in range(len(segment_value)):
            if segment_value[i] == 0:
                zero_points.append((segment_time[i], 0.0))
                if i + 1 < len(segment_value):
                    zero_points.append((segment_time[i + 1], 0.0))

        all_points = list(zip(new_xs, new_ys)) + zero_points
        all_points.sort(key=lambda p: p[0])

        if all_points:
            new_xs = np.array([p[0] for p in all_points])
            new_ys = np.array([p[1] for p in all_points])
        else:
            new_xs = np.array([])
            new_ys = np.array([])

        return new_xs, new_ys

    def simplify(time, value, max_segment_length=100):
        all_xs = []
        all_ys = []
        start_idx = 0

        while start_idx < len(time):
            end_idx = min(start_idx + max_segment_length, len(time))

            if end_idx < len(time):
                found_zero = False
                for i in range(end_idx, len(time)):
                    if value[i] == 0:
                        end_idx = i + 1
                        found_zero = True
                        break
                if not found_zero:
                    end_idx = len(time)

            segment_time = time[start_idx:end_idx]
            segment_value = value[start_idx:end_idx]

            if len(segment_time) < 2:
                start_idx = end_idx
                continue

            time_range = segment_time[-1] - segment_time[0]
            if time_range > 0:
                dense_time = np.linspace(segment_time[0], segment_time[-1], int(density * time_range))
            else:
                dense_time = segment_time

            idx = np.searchsorted(segment_time, dense_time, side="right") - 1
            idx[idx < 0] = 0
            dense_value = segment_value[idx]

            # Add original pedal event points to guarantee RDP considers them
            combined_time = np.concatenate([dense_time, segment_time])
            combined_value = np.concatenate([dense_value, segment_value])
            sort_indices = np.argsort(combined_time)
            combined_time = combined_time[sort_indices]
            combined_value = combined_value[sort_indices]

            points = np.column_stack((combined_time, combined_value))
            simplified = rdp(points, epsilon)
            xs, ys = improve_zero_values(simplified[:, 0], simplified[:, 1], segment_time, segment_value)

            if start_idx > 0 and len(all_xs) > 0 and len(xs) > 0:
                if all_xs[-1] == xs[0]:
                    xs = xs[1:]
                    ys = ys[1:]

            all_xs.extend(xs)
            all_ys.extend(ys)

            start_idx = end_idx - 1 if (end_idx > 0 and end_idx - 1 < len(value) and value[end_idx - 1] == 0) else end_idx

        return np.array(all_xs), np.array(all_ys)

    def pedal_func(events, times, x):
        i = bisect.bisect_right(times, x) - 1
        if i < 0:
            return events[0].value
        return events[i].value

    def objective(amount, events, times, x_start, y_start, x_end, y_end):
        seg = segment(x_start, y_start, x_end, y_end, amount)
        idx_start = bisect.bisect_left(times, x_start)
        idx_end = bisect.bisect_right(times, x_end)
        events_in_range = [PedalEvent(x_start, pedal_func(events, times, x_start))]
        events_in_range.extend(events[idx_start: idx_end])
        events_in_range.append(PedalEvent(x_end, pedal_func(events, times, x_end)))

        total_error = 0
        for i, e in enumerate(events_in_range[0 : -1]):
            error = seg.Lp_distance_on_const(e.value, 2, e.time, events_in_range[i+1].time)
            total_error += error

        return total_error ** (1/2)

    def fit_single_segment(events, times, x_start, y_start, x_end, y_end):
        if np.abs(y_end - y_start) < 0.1:
            return 0.5

        result = minimize_scalar(
            lambda a: objective(a, events, times, x_start, y_start, x_end, y_end),
            bounds=(-0.5, 2.5),
            method='bounded',
            options={'maxiter': 200}
        )

        return result.x

    assert len(pedal_events) > 3, "Too few pedal Events"
    events = sorted(pedal_events, key=lambda e: e.time)
    times = [e.time for e in events]

    time = np.array([e.time for e in pedal_events])
    value = np.array([e.value for e in pedal_events])
    xs, ys = simplify(time, value)

    segments = []
    for i in range(len(xs) - 1):
        x_start, y_start = xs[i], ys[i]
        x_end, y_end = xs[i+1], ys[i+1]

        amount = fit_single_segment(events, times, x_start, y_start, x_end, y_end)
        segments.append(SegmentEvent(x_start, y_start, x_end, y_end, amount))

    return segments, xs, ys


if __name__ == "__main__":
    midi_path = "maestro-v3.0.0/2015/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_12_R1_2015_wav--3.midi"
    density = 20

    print(f"Loading track: {midi_path}")
    pedal_events, note_events = get_track(midi_path)

    print(f"Processing with density={density} points/second...")
    segments, simplified_xs, simplified_ys = pedals_to_segs_with_density(pedal_events, epsilon=0.12, density=density)

    print(f"Original pedal events: {len(pedal_events)}")
    print(f"Simplified points: {len(simplified_xs)}")
    print(f"Final segments: {len(segments)}")

    fig = go.Figure()

    pedal_times = [e.time for e in pedal_events]
    pedal_values = [e.value for e in pedal_events]
    fig.add_trace(go.Scatter(
        x=pedal_times,
        y=pedal_values,
        mode='lines',
        name='Original Pedal Data',
        line=dict(color='black', width=1.5, shape='hv'),
        opacity=1
    ))

    fig.add_trace(go.Scatter(
        x=simplified_xs,
        y=simplified_ys,
        mode='markers',
        name=f'RDP Simplified (density={density})',
        marker=dict(size=6, color='red')
    ))

    seg_times, seg_values = segs_to_curve(segments, num_points=15000)
    fig.add_trace(go.Scatter(
        x=seg_times,
        y=seg_values,
        mode='lines',
        name='Final Segments',
        line=dict(color='blue', width=1.5)
    ))

    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        title=f'RDP Density Test (density={density} points/sec)',
        xaxis_title='Time (s)',
        yaxis_title='Pedal Value',
        hovermode='x unified',
        height=600
    )

    fig.show()
