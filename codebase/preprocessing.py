import numpy as np
import pretty_midi
import json
import bisect
import math
import random
from typing import List, Tuple
from copy import deepcopy
from pathlib import Path
from scipy.optimize import minimize_scalar
from .data import PedalEvent, Note, SegmentEvent, SegmentToken


class segment:
    def __init__(self, x_start, y_start, x_end, y_end, amount):
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.amount = amount * 3 - 0.5

    def a(self, x):
        return x**3

    def P1(self, x):
        assert 0 <= x <= 1
        if x < 0.5:
            return 0.5 * (2*x)**(self.a(1/self.amount))
        return 1 - 0.5 * (2*(1-x))**(self.a(1/self.amount))

    def P2(self, x):
        assert 0 <= x <= 1
        if x < 0.5:
            return 0.5 * (1 - (1-2*x)**self.a(self.amount))
        return 0.5 + 0.5 * (2*x - 1)**self.a(self.amount)

    def Plateau(self, x):
        return 0.3 * self.P1(x) + 0.7 * self.P2(x)

    def S(self, x):
        assert self.amount < 1
        return 0.5 * (math.tanh(5*(1-self.amount) * (2*x - 1)) / math.tanh(5*(1-self.amount)) + 1)

    def Double(self, x):
        if self.amount >= 1:
            return self.Plateau(x)
        return self.S(x)

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

    def Lp_distance_on_const(self, const, p, a, b):
        assert self.x_start <= a <= b <= self.x_end, "integration out of bounds"

        if (b-a) < 0.1:
            x0, x1, x2 = a, (a+b)/2, b
            f0 = abs(self(x0) - const)**p
            f1 = abs(self(x1) - const)**p
            f2 = abs(self(x2) - const)**p
            return (b - a) / 6 * (f0 + 4*f1 + f2)
        else:
            x_axis = np.linspace(a, b, int(100 * (b-a)))
            y_axis = np.array([self(x) for x in x_axis])
            approx = (y_axis - const)**p
            return approx.sum() * (b-a)


def pedals_to_segs(pedal_events: List[PedalEvent], epsilon: float = 0.12, p: int = 2) -> List[SegmentEvent]:
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
                dense_time = np.linspace(segment_time[0], segment_time[-1], int(100 * time_range))
            else:
                dense_time = segment_time

            idx = np.searchsorted(segment_time, dense_time, side="right") - 1
            idx[idx < 0] = 0
            dense_value = segment_value[idx]

            points = np.column_stack((dense_time, dense_value))
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
            error = seg.Lp_distance_on_const(e.value, p, e.time, events_in_range[i+1].time)
            total_error += error

        return total_error ** (1/p)

    def fit_single_segment(events, times, x_start, y_start, x_end, y_end):
        if np.abs(y_end - y_start) < 0.1:
            return 1

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

    return segments


def get_paths(split: str = "train", dataset_path: str = "maestro-v3.0.0", nocturnes: bool = False) -> List[str]:
    if nocturnes:
        nocturne_paths = [
            "2015/MIDI-Unprocessed_R1_D1-9-12_mid--AUDIO-from_mp3_12_R1_2015_wav--3.midi",
            "2011/MIDI-Unprocessed_22_R2_2011_MID--AUDIO_R2-D5_10_Track10_wav.midi",
            "2015/MIDI-Unprocessed_R1_D2-21-22_mid--AUDIO-from_mp3_22_R1_2015_wav--4.midi",
            "2017/MIDI-Unprocessed_058_PIANO058_MID--AUDIO-split_07-07-17_Piano-e_2-02_wav--4.midi",
            "2011/MIDI-Unprocessed_02_R2_2011_MID--AUDIO_R2-D1_03_Track03_wav.midi",
            "2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_01_R1_2015_wav--2.midi",
            "2015/MIDI-Unprocessed_R1_D1-1-8_mid--AUDIO-from_mp3_04_R1_2015_wav--3.midi",
            "2017/MIDI-Unprocessed_072_PIANO072_MID--AUDIO-split_07-08-17_Piano-e_1-06_wav--3.midi",
            "2009/MIDI-Unprocessed_06_R1_2009_03-07_ORIG_MID--AUDIO_06_R1_2009_06_R1_2009_05_WAV.midi",
            "2011/MIDI-Unprocessed_06_R3_2011_MID--AUDIO_R3-D3_05_Track05_wav.midi",
            "2009/MIDI-Unprocessed_02_R1_2009_03-06_ORIG_MID--AUDIO_02_R1_2009_02_R1_2009_03_WAV.midi",
            "2014/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_03_R1_2014_wav--4.midi",
            "2008/MIDI-Unprocessed_16_R1_2008_01-04_ORIG_MID--AUDIO_16_R1_2008_wav--3.midi",
            "2014/MIDI-UNPROCESSED_04-05_R1_2014_MID--AUDIO_05_R1_2014_wav--6.midi",
            "2017/MIDI-Unprocessed_057_PIANO057_MID--AUDIO-split_07-07-17_Piano-e_1-07_wav--3.midi",
            "2006/MIDI-Unprocessed_19_R1_2006_01-07_ORIG_MID--AUDIO_19_R1_2006_06_Track06_wav.midi",
            "2006/MIDI-Unprocessed_19_R1_2006_01-07_ORIG_MID--AUDIO_19_R1_2006_07_Track07_wav.midi",
            "2013/ORIG-MIDI_03_7_8_13_Group__MID--AUDIO_18_R2_2013_wav--3.midi",
            "2008/MIDI-Unprocessed_17_R2_2008_01-04_ORIG_MID--AUDIO_17_R2_2008_wav--1.midi"
        ]
        return [str(Path(dataset_path) / p) for p in nocturne_paths]

    json_path = Path(dataset_path) / 'maestro-v3.0.0.json'
    with open(json_path, 'r') as f:
        metadata = json.load(f)

    paths = []
    for idx in metadata['split'].keys():
        if metadata['split'][idx] == split:
            midi_path = metadata['midi_filename'][idx]
            paths.append(str(Path(dataset_path) / midi_path))

    return paths


def get_track(midi_path: str) -> Tuple[List[PedalEvent], List[Note]]:
    midi = pretty_midi.PrettyMIDI(midi_path)

    pedals = [c for c in midi.instruments[0].control_changes if c.number == 64]
    pedal_events = [PedalEvent(p.time, p.value / 127) for p in pedals]

    notes = sorted(midi.instruments[0].notes, key=lambda n: n.start)
    note_events = [Note(n.start, n.end - n.start, n.pitch - 20, 0 if n.velocity == 0 else ((n.velocity - 1) // 8) + 1) for n in notes]

    return pedal_events, note_events


def augment(
    notes: List[Note],
    segments: List[SegmentEvent],
    time_stretch_range: Tuple[float, float] = (0.9, 1.1),
) -> List[Tuple[List[Note], List[SegmentEvent]]]:

    def has_out_of_bounds(notes_list, semitones):
        return any(
            note.pitch + semitones < 0 or
            note.pitch + semitones > 87
            for note in notes_list
        )

    chosen_pitches = []
    available_pitches = list(range(-12, 12))

    while len(chosen_pitches) < 10 and available_pitches:
        n = random.choice(available_pitches)
        if has_out_of_bounds(notes, n):
            if n > 0:
                available_pitches = [p for p in available_pitches if p < n]
            else:
                available_pitches = [p for p in available_pitches if p > n]
            continue
        chosen_pitches.append(n)
        available_pitches.remove(n)

    results = []
    for semitones in chosen_pitches:
        transposed_notes = [
            Note(note.start, note.duration, note.pitch + semitones, note.velocity)
            for note in notes
        ]

        stretch_factor = random.uniform(time_stretch_range[0], time_stretch_range[1])

        stretched_notes = [
            Note(note.start * stretch_factor, note.duration * stretch_factor, note.pitch, note.velocity)
            for note in transposed_notes
        ]
        stretched_segments = [
            SegmentEvent(seg.x_start * stretch_factor, seg.y_start, seg.x_end * stretch_factor, seg.y_end, seg.amount)
            for seg in segments
        ]

        results.append((stretched_notes, stretched_segments))

    return results


def tokenize_segs(segments: List[SegmentEvent]) -> List[SegmentToken]:
    tokens = []
    tokens.append(SegmentToken(
        height=segments[0].y_start,
        amount=0,
        time=segments[0].x_start
    ))
    for seg in segments:
        token = SegmentToken(
            height=seg.y_end,
            amount=seg.amount,
            time=seg.x_end
        )
        tokens.append(token)
    return tokens


def _process_single_track(args):
    idx, path, seg_fit_tightness = args
    try:
        pedal_events, note_events = get_track(path)
        segments = pedals_to_segs(pedal_events, epsilon=seg_fit_tightness)
        augmented_tracks = augment(note_events, segments)

        results = []
        for aug_notes, aug_segments in augmented_tracks:
            aug_tokens = tokenize_segs(aug_segments)
            results.append((aug_notes, aug_tokens))

        return idx, results, None
    except Exception as e:
        return idx, None, str(e)


def create_dataset(
    split: str = "train",
    dataset_path: str = "maestro-v3.0.0",
    output_file: str = "dataset",
    seg_fit_tightness: float = 0.12,
    nocturnes: bool = False,
    track_idx: int = None,
    num_workers: int = 1,
    max_chunk_size_mb: float = 80
):
    import pickle
    import os

    print(f"Loading paths...")
    paths = get_paths(split, dataset_path, nocturnes)

    if track_idx is not None:
        paths = [paths[track_idx]]

    print(f"Found {len(paths)} tracks")

    output_dir = Path("saves") / output_file
    output_dir.mkdir(parents=True, exist_ok=True)

    max_chunk_bytes = max_chunk_size_mb * 1024 * 1024
    chunk_idx = 0
    current_chunk_path = output_dir / f"chunk_{chunk_idx}.pkl"
    chunk_file = open(current_chunk_path, 'wb')
    chunk_tracks = 0
    total_augmented = 0

    def dump_track(track_results):
        nonlocal chunk_idx, chunk_file, chunk_tracks, current_chunk_path

        for track_data in track_results:
            pickle.dump(track_data, chunk_file)
            chunk_tracks += 1

        current_size = os.path.getsize(current_chunk_path)
        if current_size >= max_chunk_bytes:
            chunk_file.close()
            print(f"Completed chunk_{chunk_idx}.pkl ({chunk_tracks} tracks, {current_size / 1024 / 1024:.1f}MB)")
            chunk_idx += 1
            chunk_tracks = 0
            current_chunk_path = output_dir / f"chunk_{chunk_idx}.pkl"
            chunk_file = open(current_chunk_path, 'wb')

        return len(track_results)

    if num_workers == 1:
        for idx, path in enumerate(paths):
            print(f"Processing track {idx+1}/{len(paths)}: {Path(path).name}")
            pedal_events, note_events = get_track(path)
            segments = pedals_to_segs(pedal_events, epsilon=seg_fit_tightness)
            augmented_tracks = augment(note_events, segments)

            track_results = []
            for aug_notes, aug_segments in augmented_tracks:
                aug_tokens = tokenize_segs(aug_segments)
                track_results.append((aug_notes, aug_tokens))

            num_augmented = dump_track(track_results)
            total_augmented += num_augmented
    else:
        from multiprocessing import Pool, cpu_count
        import time

        if num_workers is None or num_workers == -1:
            num_workers = cpu_count()
        elif num_workers <= 0:
            num_workers = max(1, cpu_count() + num_workers)

        print(f"Using {num_workers} parallel workers")
        args_list = [(idx, path, seg_fit_tightness) for idx, path in enumerate(paths)]

        failed_tracks = []
        completed = 0
        start_time = time.time()

        with Pool(processes=num_workers) as pool:
            for idx, track_results, error in pool.imap_unordered(_process_single_track, args_list):
                completed += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                eta = avg_time * (len(paths) - completed)

                if error is not None:
                    failed_tracks.append((idx, paths[idx], error))
                    print(f"[{completed}/{len(paths)}] FAILED: {Path(paths[idx]).name} - {error}")
                else:
                    num_augmented = dump_track(track_results)
                    total_augmented += num_augmented
                    print(f"[{completed}/{len(paths)}] {Path(paths[idx]).name} | "
                          f"{num_augmented} augmented versions | Avg: {avg_time:.1f}s/track | ETA: {eta/60:.1f}min")

        if failed_tracks:
            print(f"\nWarning: {len(failed_tracks)} tracks failed")

    chunk_file.close()
    final_size = os.path.getsize(current_chunk_path)
    if final_size > 0:
        print(f"Completed chunk_{chunk_idx}.pkl ({chunk_tracks} tracks, {final_size / 1024 / 1024:.1f}MB)")

    print(f"\nDataset creation complete! Saved {len(paths)} tracks with {total_augmented} total augmented versions across {chunk_idx + 1} chunks in {output_dir}")
