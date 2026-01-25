import plotly.graph_objects as go
import numpy as np
from dataclasses import dataclass


@dataclass
class SegmentEvent:
    x_start: float
    y_start: float
    x_end: float
    y_end: float
    amount: float


@dataclass
class Note:
    start: float
    duration: float
    pitch: int
    velocity: int


class Segment:
    def __init__(self, x_start, y_start, x_end, y_end, amount):
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.amount = amount * 3 - 0.5

    def a(self, x):
        return x**3

    def S1(self, x):
        return x**self.a(self.amount)

    def S2(self, x):
        return 1 - (1-x)**self.a(2-self.amount)

    def Single(self, x):
        if self.amount > 1:
            return self.S1(x)
        return self.S2(x)

    def norm(self, x):
        return (x - self.x_start) / (self.x_end - self.x_start)

    def scale(self, x):
        return self.y_start + x * (self.y_end - self.y_start)

    def __call__(self, x):
        return self.scale(self.Single(self.norm(x)))


def segs_to_curve(segments, num_points=1000):
    min_time = segments[0].x_start
    max_time = segments[-1].x_end

    times = np.linspace(min_time, max_time, num_points)
    values = np.zeros(num_points)

    for seg_event in segments:
        seg = Segment(seg_event.x_start, seg_event.y_start, seg_event.x_end, seg_event.y_end, seg_event.amount)
        mask = (times >= seg.x_start) & (times <= seg.x_end)
        values[mask] = [seg(t) for t in times[mask]]

    return times, values


def plot_embedding_example(segments, notes):
    times, values = segs_to_curve(segments)

    fig = go.Figure()

    endpoints = [
        (segments[0].x_start, segments[0].y_start),
        (segments[0].x_end, segments[0].y_end),
        (segments[1].x_end, segments[1].y_end),
    ]
    seg1_mask = times <= segments[0].x_end
    seg2_mask = times >= segments[1].x_start

    # Point 1
    fig.add_trace(go.Scatter(
        x=[endpoints[0][0]], y=[endpoints[0][1]],
        mode='markers',
        marker=dict(size=10, color='orange'),
        name=f'Point 1: (t={endpoints[0][0]}, h={endpoints[0][1]})'
    ))
    # Amount 1 (segment 1 curve)
    fig.add_trace(go.Scatter(
        x=times[seg1_mask], y=values[seg1_mask],
        mode='lines',
        line=dict(color='blue', width=2),
        name=f'Amount 1: {segments[0].amount}'
    ))
    # Point 2
    fig.add_trace(go.Scatter(
        x=[endpoints[1][0]], y=[endpoints[1][1]],
        mode='markers',
        marker=dict(size=10, color='orange'),
        name=f'Point 2: (t={endpoints[1][0]}, h={endpoints[1][1]})'
    ))
    # Amount 2 (segment 2 curve)
    fig.add_trace(go.Scatter(
        x=times[seg2_mask], y=values[seg2_mask],
        mode='lines',
        line=dict(color='blue', width=2),
        name=f'Amount 2: {segments[1].amount}'
    ))
    # Point 3
    fig.add_trace(go.Scatter(
        x=[endpoints[2][0]], y=[endpoints[2][1]],
        mode='markers',
        marker=dict(size=10, color='orange'),
        name=f'Point 3: (t={endpoints[2][0]}, h={endpoints[2][1]})'
    ))

    # Notes as rectangles
    for i, note in enumerate(notes):
        end = note.start + note.duration
        fig.add_shape(
            type='rect',
            x0=note.start, x1=end,
            y0=note.pitch / 100, y1=(note.pitch + 5) / 100,
            fillcolor='rgba(144, 238, 144, 1)',
            line=dict(width=0),
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='rgba(144, 238, 144, 1)', symbol='square'),
            name=f'Note {i+1}: (s={note.start}, d={note.duration}, p={note.pitch}, v={note.velocity})'
        ))

    fig.update_layout(
        title="Note and Segment tokenization example",
        xaxis_title="Time",
        yaxis_title="Value",
        xaxis=dict(range=[-0.1, 2.1]),
        yaxis=dict(range=[-0.05, 1.05]),
        width=800,
        height=400,
        legend=dict(x=1.02, y=1),
    )

    return fig


def print_tokenizations(notes, segments):
    a_parts = []
    for note in notes:
        midi_pitch = note.pitch + 20
        midi_vel = note.velocity * 8
        a_parts.append(f"ON({midi_pitch}), T({note.start}), Vel({midi_vel})")
    for note in notes:
        midi_pitch = note.pitch + 20
        end_time = note.start + note.duration
        a_parts.append(f"OFF({midi_pitch}), T({end_time})")
    a = ", ".join(a_parts)

    b_parts = [f"({n.start}, {n.duration}, {n.pitch}, {n.velocity})" for n in notes]
    b = ", ".join(b_parts)

    c_parts = [f"Val({segments[0].y_start}), T({segments[0].x_start})"]
    for seg in segments:
        c_parts.append(f"Amt({seg.amount}), Val({seg.y_end}), T({seg.x_end})")
    c = ", ".join(c_parts)

    d_parts = [f"({segments[0].y_start}, 0, {segments[0].x_start})"]
    for seg in segments:
        d_parts.append(f"({seg.y_end}, {seg.amount}, {seg.x_end})")
    d = ", ".join(d_parts)

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = {c}")
    print(f"d = {d}")


if __name__ == "__main__":
    segments = [
        SegmentEvent(x_start=0, y_start=0.5, x_end=0.7, y_end=0.9, amount=0.6),
        SegmentEvent(x_start=0.7, y_start=0.9, x_end=2.0, y_end=0, amount=0.8),
    ]
    notes = [
        Note(start=0.2, duration=0.6, pitch=30, velocity=11),
        Note(start=0.5, duration=1.0, pitch=55, velocity=9),
    ]

    print_tokenizations(notes, segments)
    fig = plot_embedding_example(segments, notes)
    fig.show()
