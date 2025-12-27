import numpy as np
from codebase.utils import load_pkl

dataset = load_pkl('small.pkl')

all_dts = []
all_times = []

print(f"Dataset has {len(dataset.tracks)} tracks\n")

for track_idx, (notes, tokens) in enumerate(dataset.tracks[:5]):
    print(f"\nTrack {track_idx}: {len(tokens)} tokens")
    dts = []
    for i in range(1, len(tokens)):
        dt = tokens[i].time - tokens[i-1].time
        dts.append(dt)
        all_dts.append(dt)
        all_times.append(tokens[i].time)

    if dts:
        print(f"  dt min={min(dts):.4f}, max={max(dts):.4f}, mean={np.mean(dts):.4f}, median={np.median(dts):.4f}")
        print(f"  First 10 dts: {[f'{dt:.4f}' for dt in dts[:10]]}")

print(f"\n{'='*60}")
print(f"OVERALL STATISTICS (all tracks):")
print(f"  Total tokens: {len(all_dts)}")
print(f"  dt min: {min(all_dts):.4f}")
print(f"  dt max: {max(all_dts):.4f}")
print(f"  dt mean: {np.mean(all_dts):.4f}")
print(f"  dt median: {np.median(all_dts):.4f}")
print(f"  dt std: {np.std(all_dts):.4f}")
print(f"\nPercentiles:")
print(f"  10th: {np.percentile(all_dts, 10):.4f}")
print(f"  25th: {np.percentile(all_dts, 25):.4f}")
print(f"  50th: {np.percentile(all_dts, 50):.4f}")
print(f"  75th: {np.percentile(all_dts, 75):.4f}")
print(f"  90th: {np.percentile(all_dts, 90):.4f}")
print(f"  95th: {np.percentile(all_dts, 95):.4f}")
print(f"  99th: {np.percentile(all_dts, 99):.4f}")

print(f"\nHistogram of dts:")
hist, bins = np.histogram(all_dts, bins=[0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 10.0])
for i in range(len(hist)):
    print(f"  {bins[i]:.2f} - {bins[i+1]:.2f}: {hist[i]} ({100*hist[i]/len(all_dts):.1f}%)")