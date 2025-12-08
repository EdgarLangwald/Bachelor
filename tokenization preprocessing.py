import numpy as np
from typing import List, Tuple, Dict
import pretty_midi
from dataclasses import dataclass
import pickle
import json
import math
import bisect
from pathlib import Path
from abc import ABC, abstractmethod
from scipy.optimize import minimize_scalar

@dataclass
class NoteEvent:
    start: float
    end: float
    pitch: float
    velocity: float

@dataclass
class PedalEvent:
    time: float
    value: float

@dataclass
class SegmentEvent:
    # curve_type: str
    x_start: float
    y_start: float
    x_end: float
    y_end: float
    amount: float

class segment:
    def __init__(self, x_start, y_start, x_end, y_end, amount):
        # assert curve_type in ['single', 'double', 'hold'], "curve_type must be 'single', 'double' or 'hold'"
        # self.curve_type = curve_type
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.amount = amount

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

    '''def raw_curve(self, x):
        if self.curve_type == 'hold':
            return self.y_start
        elif self.curve_type == 'single':
            return self.Single(x)
        else:
            return self.Double(x)'''

    def norm(self, x):
        return (x - self.x_start) / (self.x_end - self.x_start)

    def scale(self, x):
        return self.y_start + x * (self.y_end - self.y_start)

    def __call__(self, x):
        return self.scale(self.Single(self.norm(x))) ### Change to Simple

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


def rdp_simplify(time, value, epsilon, max_segment_length=500):
    """
    Reduces points while preserving shape using perpendicular distance threshold.
    Splits data into segments at zero-crossings to handle large datasets efficiently.
    epsilon: tolerance (higher = more aggressive reduction)
    max_segment_length: minimum number of events per segment before searching for a split
    """
    def perpendicular_distance(point, line_start, line_end):
        if np.array_equal(line_start, line_end):
            return np.linalg.norm(point - line_start)
        return np.abs(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)

    def rdp(points, epsilon):
        dmax = 0
        index = 0
        end = len(points) - 1

        for i in range(1, end):
            d = perpendicular_distance(points[i], points[0], points[end])
            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            left = rdp(points[:index+1], epsilon)
            right = rdp(points[index:], epsilon)
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


class FitHandler:
    def __init__(self, pedal_events: List[PedalEvent]):
        assert len(pedal_events) > 3, "Too few pedal Events"
        self.events = sorted(pedal_events, key=lambda e: e.time)
        self.times = [e.time for e in self.events]

    def pedal_func(self, x):
        i = bisect.bisect_right(self.times, x) - 1
        if i < 0:
            return self.events[0].value
        return self.events[i].value

    def _objective(self, amount, x_start, y_start, x_end, y_end, p):
        seg = segment(x_start, y_start, x_end, y_end, amount)

        idx_start = bisect.bisect_left(self.times, x_start)
        idx_end = bisect.bisect_right(self.times, x_end)
        events_in_range = [PedalEvent(x_start, self.pedal_func(x_start))]
        events_in_range.extend(self.events[idx_start: idx_end])
        events_in_range.append(PedalEvent(x_end, self.pedal_func(x_end)))

        total_error = 0
        for i, e in enumerate(events_in_range[0 : -1]):
            error = seg.Lp_distance_on_const(e.value, p, e.time, events_in_range[i+1].time)
            total_error += error

        return total_error ** (1/p)

    def fit_segment(self, x_start, y_start, x_end, y_end, p=2) -> float:
        from scipy.optimize import minimize_scalar

        if np.abs(y_end - y_start) < 0.1:
            return 1

        result = minimize_scalar(
            lambda a: self._objective(a, x_start, y_start, x_end, y_end, p),
            bounds=(-0.5, 2.5),
            method='bounded',
            options={'maxiter': 200}
        )

        return result.x

    def fit_curve(self, x_values, y_values, p=2, get_raw_params=False) -> List[object]:
        segments = []

        for i in range(len(x_values) - 1):
            x_start, y_start = x_values[i], y_values[i]
            x_end, y_end = x_values[i+1], y_values[i+1]

            result = self.fit_segment(x_start, y_start, x_end, y_end, p)

            amount = result
            if not get_raw_params:
                segments.append(segment(x_start, y_start, x_end, y_end, amount))
            else:
                segments.append(SegmentEvent(x_start, y_start, x_end, y_end, amount))

        return segments


class MIDIPreprocessor(ABC):
    def __init__(self):
        self.path_to_pickle = str(None)

    @abstractmethod
    def get_paths(self) -> List:
        pass

    @abstractmethod
    def preprocess_data(self, paths) -> object:
        pass

    def safe_to_pickle(self, preprocessed_data, file_name: str):
        self.path_to_pickle = Path(file_name)
        with open(self.path_to_pickle, 'wb') as f:
            pickle.dump(preprocessed_data, f)

    def load_from_pickle(self) -> object:
        with open(self.path_to_pickle, "rb") as f:
            data = pickle.load(f)
        return data


class SingleSplinePreprocessor(MIDIPreprocessor):
    def __init__(self):
        super().__init__()

    def get_paths(self, track) -> str:
        json_path = "maestro-v3.0.0/maestro-v3.0.0.json"
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        train_paths = []
        for idx in metadata['split'].keys():
            if metadata['split'][idx] == "train":
                midi_path = metadata['midi_filename'][idx]
                train_paths.append(midi_path)

        return train_paths[track]

    def get_pedal(self, path) -> List[PedalEvent]:
        single_file = pretty_midi.PrettyMIDI(path)
        pedals = single_file.instruments[0].control_changes
        pedals = [pedals[i] for i in range(len(pedals)) if pedals[i].number == 64]

        pedal_events = []
        for i in range(len(pedals)):
            pedal_events.append(PedalEvent(
                time=pedals[i].time,
                value=pedals[i].value/127
            ))

        return pedal_events

    def preprocess_data(self, paths) -> object:
        return 1


class RDPCustomPreprocessor(MIDIPreprocessor):
    def __init__(self):
        super().__init__()

    def get_paths(self, split=None, nocturnes=False, track=None):
        if split is not None or track is not None:
            if split is None:
                split = "train"
            else:
                assert split in ["train", "test", "validation"], "split can only be 'train', 'test' or 'validation'"
            json_path = "maestro-v3.0.0/maestro-v3.0.0.json"
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            paths = []
            for idx in metadata['split'].keys():
                if metadata['split'][idx] == split:
                    midi_path = metadata['midi_filename'][idx]
                    paths.append(midi_path)
            if track is not None:
                return paths[track]
            return paths
        if nocturnes:
            paths = [
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
        

    def get_pedal(self, path) -> List[PedalEvent]:
        single_file = pretty_midi.PrettyMIDI(path)
        pedals = single_file.instruments[0].control_changes
        pedals = [pedals[i] for i in range(len(pedals)) if pedals[i].number == 64]

        pedal_events = []
        for i in range(len(pedals)):
            pedal_events.append(PedalEvent(
                time=pedals[i].time,
                value=pedals[i].value/127
            ))

        return pedal_events

    def get_raw_events(self, paths, smoothing_factor=0.12, p=2, max_segment_length=500, verbose=False) -> Tuple[List[List[SegmentEvent]], List[List[NoteEvent]]]:

        all_segments = []

        for idx, path in enumerate(paths):
            if verbose:
                print(f"Processing track {idx+1}/{len(paths)}: {path}")

            pedal_events = self.get_pedal("maestro-v3.0.0/" + path)

            time = np.array([e.time for e in pedal_events])
            value = np.array([e.value for e in pedal_events])

            xs, ys = rdp_simplify(time, value, smoothing_factor, max_segment_length)

            fh = FitHandler(pedal_events)
            fitted_segments = fh.fit_curve(xs, ys, p=p, get_raw_params=True)

            all_segments.append(fitted_segments)

            if verbose:
                print(f"  -> {len(fitted_segments)} segments fitted to {len(pedal_events)} pedal events")

        return all_segments, self.get_notes(paths)
    
    def get_notes(self, paths) -> List[List[NoteEvent]]:
        all_events = []
        for path in paths:
            path = "maestro-v3.0.0/" + path
            file = pretty_midi.PrettyMIDI(path)
            notes = file.instruments[0].notes
            notes = sorted(notes, key=lambda note: note.start)

            note_events = [
                NoteEvent(
                    start=note.start,
                    end=note.end,
                    pitch=(note.pitch - 20),
                    velocity=note.velocity
                )
                for note in notes
            ]
            all_events.append(note_events)

        return all_events
        
    def tokenize(self, events, type):
        assert type in ["CC", "CT", "TC", "TT"], 'type must be one of "CC", "CT", "TC" or "TT"'

        all_segments, all_notes = events

        tokenized_notes = []
        tokenized_segments = []

        amount_boundaries = [
            -0.50, 0.01, 0.35, 0.60, 0.79, 0.93,
            1.07, 1.21, 1.40, 1.65, 1.99, 2.50
        ]

        def get_amount_bucket(amount):
            for i in range(len(amount_boundaries) - 1):
                if amount_boundaries[i] <= amount < amount_boundaries[i + 1]:
                    return i
            return len(amount_boundaries) - 2

        def get_value_bucket(value):
            if value == 0:
                return 0
            elif value == 1:
                return 21
            else:
                bucket = int((value - 0.0001) / 0.05) + 1
                return min(bucket, 20)

        # Tokenize notes
        for notes in all_notes:
            if type in ["CC", "CT"]:  # Continuous time for notes
                tokens = []
                for note in notes:
                    pitch_token = int(note.pitch)
                    vel_token = (0 if note.velocity == 0 else ((note.velocity - 1) // 4 + 1)) + 176
                    off_token = pitch_token + 88

                    tokens.append((pitch_token, note.start))
                    tokens.append((vel_token, note.start))
                    tokens.append((off_token, note.end))

                tokens.sort(key=lambda x: x[1])
                tokenized_notes.append(tokens)

            else:  # Tokenized time for notes (TC, TT)
                events_list = []

                for note in notes:
                    pitch_token = int(note.pitch)
                    vel_token = (0 if note.velocity == 0 else ((note.velocity - 1) // 4 + 1)) + 176
                    off_token = pitch_token + 88

                    events_list.append((note.start, 'on', pitch_token, vel_token))
                    events_list.append((note.end, 'off', off_token))

                events_list.sort(key=lambda x: x[0])

                tokens = []
                for event in events_list:
                    time_token = int(event[0] / 0.025)

                    if event[1] == 'on':
                        tokens.append(event[2] + 4000)
                        tokens.append(time_token)
                        tokens.append(event[3] + 4000)
                    else:
                        tokens.append(event[2] + 4000)
                        tokens.append(time_token)

                tokenized_notes.append(tokens)

        # Tokenize segments
        for segments in all_segments:
            if type in ["CC", "TC"]:  # Continuous time for segments
                tokens = []
                for seg in segments:
                    val_token = get_value_bucket(seg.y_start)
                    amt_token = get_amount_bucket(seg.amount) + 22

                    tokens.append((val_token, seg.x_start))
                    tokens.append((amt_token, seg.x_start))

                tokenized_segments.append(tokens)

            else:  # Tokenized time for segments (CT, TT)
                tokens = []
                for seg in segments:
                    val_token = get_value_bucket(seg.y_start)
                    amt_token = get_amount_bucket(seg.amount) + 22
                    time_token = int(seg.x_start / 0.025)

                    tokens.append(val_token + 4000)
                    tokens.append(time_token)
                    tokens.append(amt_token + 4000)

                tokenized_segments.append(tokens)

        return tokenized_notes, tokenized_segments
