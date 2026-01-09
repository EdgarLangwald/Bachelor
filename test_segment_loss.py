import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from codebase.train import compute_segment_loss, compute_param_loss
from codebase.utils import segs_to_curve
from codebase.data import SegmentEvent

target = (0, 1, 1, 1, 0.5)
predictions = [
    (0, 1, 0.01, 0, 0.5),
    (0, 1, 1, 0, 0),
    (0, 1, 1, 0, 1),
    (0, 1, 3, 1, 0.5),
    (0, 1, 1, 0.8, 0.5)
]

device = 'cpu'

x_start_gt = target[0]
y_start_gt = target[1]
x_end_gt = target[2]
y_end_gt = target[3]
amount_gt = target[4]

segment_losses = []
param_losses = []

for pred in predictions:
    x_start = pred[0]
    y_start = pred[1]
    x_end_pred = pred[2]
    y_end_pred = pred[3]
    amount_pred = pred[4]

    tokens = torch.tensor([
        [[0.0, 0.0, 0.0]],
        [[y_start, 0.0, x_start]],
        [[y_end_gt, amount_gt, x_end_gt]]
    ], device=device, dtype=torch.float32).transpose(0, 1)

    log_delta_time = torch.log(torch.tensor(x_end_pred - x_start, device=device, dtype=torch.float32))

    model_output = {
        'height': torch.tensor([
            [[0.0]],
            [[y_end_pred]],
            [[0.0]]
        ], device=device, dtype=torch.float32).transpose(0, 1),
        'amount': torch.tensor([
            [[0.0]],
            [[amount_pred]],
            [[0.0]]
        ], device=device, dtype=torch.float32).transpose(0, 1),
        'time': torch.tensor([
            [[0.0]],
            [[log_delta_time.item()]],
            [[0.0]]
        ], device=device, dtype=torch.float32).transpose(0, 1)
    }

    tgt_key_padding_mask = torch.tensor([[False, True, False]], dtype=torch.bool, device=device)

    seg_loss = compute_segment_loss(model_output, tokens, tgt_key_padding_mask)
    param_loss, _ = compute_param_loss(model_output, tokens, tgt_key_padding_mask)

    segment_losses.append(seg_loss.item())
    param_losses.append(param_loss.item())

sorted_results = sorted(
    zip(predictions, segment_losses, param_losses, range(len(predictions))),
    key=lambda x: x[1] + x[2]
)

print("Losses for each prediction (sorted by total loss):")
for pred, seg_loss, param_loss, orig_idx in sorted_results:
    total_loss = seg_loss + param_loss
    print(f"Prediction {orig_idx+1}: {total_loss:.4f} = {seg_loss:.4f} + {param_loss:.4f}")

fig = go.Figure()

target_event = SegmentEvent(
    x_start=target[0],
    y_start=target[1],
    x_end=target[2],
    y_end=target[3],
    amount=target[4]
)
times_target, values_target = segs_to_curve([target_event])
fig.add_trace(go.Scatter(
    x=times_target,
    y=values_target,
    mode='lines',
    name=f'Target {target}',
    line=dict(color='black', width=3)
))

colors = ['red', 'blue', 'green', 'orange', 'purple']

for pred, seg_loss, param_loss, orig_idx in sorted_results:
    pred_event = SegmentEvent(
        x_start=pred[0],
        y_start=pred[1],
        x_end=pred[2],
        y_end=pred[3],
        amount=pred[4]
    )
    times_pred, values_pred = segs_to_curve([pred_event])
    total_loss = seg_loss + param_loss
    color = colors[orig_idx]
    fig.add_trace(go.Scatter(
        x=times_pred,
        y=values_pred,
        mode='lines',
        name=f'Pred {orig_idx+1}: {total_loss:.4f} = {seg_loss:.4f} + {param_loss:.4f}',
        line=dict(color=color, width=2)
    ))

fig.update_layout(
    title='Segment Loss vs Parameter Loss Visualization',
    xaxis_title='x',
    yaxis_title='y',
    legend=dict(x=1.05, y=1),
    width=1200,
    height=600
)

fig.show()
