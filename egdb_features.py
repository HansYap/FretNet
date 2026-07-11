import os
from dataclasses import dataclass
import librosa
import mido
import numpy as np
from labels import build_label_array, stack_polyphonic_labels

TRACK_TO_STRING = {1: 'e', 2: 'B', 3: 'G', 4: 'D', 5: 'A', 6: 'E'}


MIN_FRET, MAX_FRET = 0, 19

TIMBRES = ['DI', 'Marshall', 'Ftwin', 'Mesa', 'JCjazz', 'Plexi']

DEFAULT_TEMPO = 500000  


@dataclass
class NoteData:
    """just to match mirdata's per string note annotation object
     .intervals (N,2) and .pitches (N,) for build_label_array() to work."""
    intervals: np.ndarray
    pitches: np.ndarray


def _extract_note_intervals(track, ticks_per_beat, tempo):
    #(onset_sec, offset_sec, midi_pitch) tuples for a single string's MIDI track
    intervals = []
    open_notes = {}
    abs_tick = 0
    for msg in track:
        abs_tick += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            open_notes[msg.note] = abs_tick
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            onset_tick = open_notes.pop(msg.note, None)
            if onset_tick is None:
                continue  
            onset_sec = mido.tick2second(onset_tick, ticks_per_beat, tempo)
            offset_sec = mido.tick2second(abs_tick, ticks_per_beat, tempo)
            intervals.append((onset_sec, offset_sec, msg.note))
    return intervals


def load_egdb_notes(midi_path, string_midi):
    """
    Parses one EGDB clip's MIDI annotation into {string_name: NoteData | None},
    the same shape GuitarSet's track.notes provides, so build_label_array()
    from labels.py can be reused completely unchanged.
    Returns (notes_by_string, n_dropped) - n_dropped is the count of notes
    whose implied fret fell outside 0-19 (pickup-bleed noise, discarded).
    """
    mid = mido.MidiFile(midi_path)

    tempo_events = [msg.tempo for msg in mid.tracks[0] if msg.type == 'set_tempo']
    tempo = tempo_events[0] if tempo_events else DEFAULT_TEMPO
    if len(tempo_events) > 1:
        print(f"WARNING======= {midi_path} has {len(tempo_events)} tempo changes, "
              f"using only the first ({mido.tempo2bpm(tempo):.1f} BPM) - "
              f"timing may drift late in this clip", flush=True)

    notes_by_string = {}
    n_dropped = 0

    for track_idx in range(1, 7):
        string_name = TRACK_TO_STRING[track_idx]
        open_midi = string_midi[string_name]
        raw = _extract_note_intervals(mid.tracks[track_idx], mid.ticks_per_beat, tempo)

        starts, ends, pitches = [], [], []
        for onset_sec, offset_sec, pitch in raw:
            fret = pitch - open_midi
            if fret < MIN_FRET or fret > MAX_FRET:
                n_dropped += 1
                continue
            starts.append(onset_sec)
            ends.append(offset_sec)
            pitches.append(pitch)

        if starts:
            notes_by_string[string_name] = NoteData(
                intervals=np.array(list(zip(starts, ends))),
                pitches=np.array(pitches, dtype=float),
            )
        else:
            notes_by_string[string_name] = None  # build_label_array treats None as "no notes"

    return notes_by_string, n_dropped


def get_egdb_clip_ids(egdb_root):
    label_dir = os.path.join(egdb_root, 'audio_label')
    return sorted(int(f.split('.')[0]) for f in os.listdir(label_dir) if f.endswith('.midi'))


def process_egdb_track(clip_id, timbre, egdb_root, string_midi, sr=22050, hop_length=512):
    audio_path = os.path.join(egdb_root, f'audio_{timbre}', f'{clip_id}.wav')
    midi_path = os.path.join(egdb_root, 'audio_label', f'{clip_id}.midi')

    y, _ = librosa.load(audio_path, sr=sr)
    cqt = librosa.cqt(y=y, sr=sr, hop_length=hop_length,
                       fmin=librosa.note_to_hz('C2'),
                       bins_per_octave=24, n_bins=96)
    cqt = librosa.amplitude_to_db(np.abs(cqt))

    notes_by_string, n_dropped = load_egdb_notes(midi_path, string_midi)

    label_arrays = {
        s: build_label_array(notes_by_string[s], open_midi, cqt.shape[1], sr, hop_length)
        for s, open_midi in string_midi.items()
    }
    poly_labels = stack_polyphonic_labels(label_arrays, string_midi)
    return cqt, poly_labels, n_dropped