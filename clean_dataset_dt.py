import pickle
import os
from pathlib import Path
from codebase.utils import load_dataset
from codebase.data import SegmentToken

def clean_tokens(tokens, threshold=0.03):
    """
    Clean tokens to ensure minimum dt separation.

    Rules:
    - If dt < 0.03 and both heights are 0: Remove both, replace with middle position at height 0
    - If dt < 0.03 and first height is nonzero and second is zero: Adjust first time to be 0.03 apart
    - If dt < 0.03 and first height is zero and second is nonzero: Adjust second time to be 0.03 apart
    """
    if len(tokens) <= 1:
        return tokens

    cleaned = []
    i = 0

    while i < len(tokens):
        if i == len(tokens) - 1:
            # Last token, just add it
            cleaned.append(tokens[i])
            i += 1
            continue

        curr = tokens[i]
        next_tok = tokens[i + 1]
        dt = next_tok.time - curr.time

        if dt < threshold:
            height_curr = curr.height
            height_next = next_tok.height

            # Case 1: Both heights are 0
            if abs(height_curr) < 1e-9 and abs(height_next) < 1e-9:
                # Replace both with middle position at height 0
                middle_time = (curr.time + next_tok.time) / 2
                middle_token = SegmentToken(
                    height=0.0,
                    amount=curr.amount,  # Keep the amount from first token
                    time=middle_time
                )
                cleaned.append(middle_token)
                print(f"  Merged two zero-height tokens at t={curr.time:.6f} and t={next_tok.time:.6f} -> t={middle_time:.6f}")
                i += 2  # Skip both tokens
                continue

            # Case 2: First height is nonzero and second is zero
            elif abs(height_curr) >= 1e-9 and abs(height_next) < 1e-9:
                # Adjust first token's time to be threshold apart from second
                adjusted_time = next_tok.time - threshold
                adjusted_token = SegmentToken(
                    height=curr.height,
                    amount=curr.amount,
                    time=adjusted_time
                )
                cleaned.append(adjusted_token)
                print(f"  Adjusted nonzero->zero: t={curr.time:.6f} -> t={adjusted_time:.6f} (next at t={next_tok.time:.6f})")
                i += 1  # Move to next token (which will be added in next iteration)
                continue

            # Case 3: First height is zero and second is nonzero
            elif abs(height_curr) < 1e-9 and abs(height_next) >= 1e-9:
                # Keep first token as is, adjust second token's time to be threshold apart
                cleaned.append(curr)
                adjusted_time = curr.time + threshold
                adjusted_token = SegmentToken(
                    height=next_tok.height,
                    amount=next_tok.amount,
                    time=adjusted_time
                )
                cleaned.append(adjusted_token)
                print(f"  Adjusted zero->nonzero: next t={next_tok.time:.6f} -> t={adjusted_time:.6f} (prev at t={curr.time:.6f})")
                i += 2  # Skip both tokens since we've handled both
                continue

            # Other cases: Keep both tokens but warn
            else:
                print(f"  [WARNING] Small dt={dt:.6f} at t={curr.time:.6f}, heights {height_curr:.3f}->{height_next:.3f} - keeping as is")
                cleaned.append(curr)
                i += 1
                continue

        # Normal case: dt >= threshold
        cleaned.append(curr)
        i += 1

    return cleaned


def clean_dataset(dataset_path, output_path=None, threshold=0.03):
    """
    Clean a dataset by ensuring minimum dt separation between tokens.

    Args:
        dataset_path: Path to input dataset (.pkl file) - relative to saves/
        output_path: Path to save cleaned dataset (defaults to dataset_path with '_cleaned' suffix)
        threshold: Minimum dt value (default 0.03)
    """
    if output_path is None:
        if dataset_path.endswith('.pkl'):
            output_path = dataset_path.replace('.pkl', '_cleaned.pkl')
        else:
            output_path = dataset_path + '_cleaned'

    output_full_path = Path("saves") / output_path

    print("="*60)
    print("Dataset Cleaning")
    print("="*60)
    print(f"Input:  {dataset_path}")
    print(f"Output: {output_path}")
    print(f"Threshold: dt < {threshold}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    print(f"Loaded {len(dataset)} tracks")
    print()

    # Clean each track
    total_issues = 0
    tracks_with_issues = 0
    negative_dt_count = 0

    cleaned_tracks = []

    for track_idx in range(len(dataset)):
        notes, tokens = dataset.tracks[track_idx]

        # Find issues in this track
        issues = []
        for j in range(1, len(tokens)):
            dt = tokens[j].time - tokens[j-1].time
            if dt < 0:
                print(f"[CRITICAL ERROR] Track {track_idx}: NEGATIVE dt={dt:.6f} at position {j-1}->{j}")
                print(f"  Time {j-1}: {tokens[j-1].time:.6f}")
                print(f"  Time {j}: {tokens[j].time:.6f}")
                negative_dt_count += 1
            if dt < threshold:
                issues.append((j-1, j, dt))

        if len(issues) > 0:
            tracks_with_issues += 1
            total_issues += len(issues)
            print(f"Track {track_idx}: {len(issues)} issue(s)")

            # Clean this track
            cleaned_tokens = clean_tokens(tokens, threshold)
            cleaned_tracks.append((notes, cleaned_tokens))

            print(f"  Original: {len(tokens)} tokens")
            print(f"  Cleaned:  {len(cleaned_tokens)} tokens")
            print()
        else:
            # No issues, keep original
            cleaned_tracks.append((notes, tokens))

    print("="*60)
    print("Summary")
    print("="*60)
    print(f"Tracks with issues: {tracks_with_issues} / {len(dataset)}")
    print(f"Total issues fixed: {total_issues}")
    print(f"NEGATIVE dt found: {negative_dt_count}")
    if negative_dt_count > 0:
        print("[CRITICAL] Dataset has REVERSED time tokens - this will cause NEGATIVE segment loss!")
    print()

    # Save cleaned dataset
    print(f"Saving cleaned dataset to {output_path}...")

    output_full_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_full_path, 'wb') as f:
        for notes, tokens in cleaned_tracks:
            pickle.dump((notes, tokens), f)

    print("Done!")
    print()

    # Verify the cleaned dataset
    print("Verifying cleaned dataset...")
    cleaned_dataset = load_dataset(output_path)

    remaining_issues = 0
    remaining_negative_dt = 0
    for track_idx in range(len(cleaned_dataset)):
        notes, tokens = cleaned_dataset.tracks[track_idx]
        for j in range(1, len(tokens)):
            dt = tokens[j].time - tokens[j-1].time
            if dt < 0:
                remaining_negative_dt += 1
            if dt < threshold:
                remaining_issues += 1

    if remaining_issues == 0 and remaining_negative_dt == 0:
        print(f"[SUCCESS] No remaining issues with dt < {threshold}")
        print(f"[SUCCESS] No negative dt values")
    else:
        if remaining_issues > 0:
            print(f"[WARNING] Still {remaining_issues} issues with dt < {threshold} remaining!")
        if remaining_negative_dt > 0:
            print(f"[CRITICAL ERROR] Still {remaining_negative_dt} NEGATIVE dt values remaining!")

    print("="*60)


if __name__ == '__main__':
    # Set your dataset path here
    dataset_path = "complete_dataset/chunk_0.pkl"

    clean_dataset(dataset_path, threshold=0.03)
