import os
from dataclasses import dataclass

import librosa
import mido
import numpy as np

from labels import build_label_array, stack_polyphonic_labels

# A 240-clip scan confirmed 7 tracks per clip in every sample checked (1 meta/tempo
# track + 6 per-string tracks, ordered high string -> low string, matching literal
# track_name '1'..'6' meta messages seen on some clips) - but that scan used
# enumerate() and silently tolerated any file with fewer tracks rather than
# verifying count==7, so it didn't catch clips where a fully-silent string's
# track gets omitted from the file entirely rather than written empty. Real data
# hit exactly that: an IndexError on a clip with <7 tracks. So track position is
# NOT used to identify a string - each present track is instead classified by
# which string's open-pitch + fret 0-19 window its notes actually fit, which is
# robust regardless of track count or which track got dropped.

# Standard tuning confirmed dataset-wide: for every string, the 1st and 5th
# pitch percentile land exactly on the open-string MIDI note (config.STRING_MIDI).
# Only 0.24% of notes (86/35673) fall outside a valid fret 0-19 - consistent
# with hexaphonic pickup bleed (the same phenomenon GuitarSet needed
# de-bleeding for), not a tuning mismatch. Those notes are dropped, not clamped,
# so they don't inject a spurious fret-0 (or fret-19) bias.
MIN_FRET, MAX_FRET = 0, 19
CLASSIFICATION_CONFIDENCE_WARN = 0.9  # print a warning if a track's best-fit string match is weaker than this

TIMBRES = ['DI', 'Marshall', 'Ftwin', 'Mesa', 'JCjazz', 'Plexi']

DEFAULT_TEMPO = 500000  # 120bpm, only used if a clip is somehow missing a tempo event


@dataclass
class NoteData:
    """Minimal stand-in for mirdata's per-string note annotation object -
    just needs .intervals (N,2) and .pitches (N,) for build_label_array() to work."""
    intervals: np.ndarray
    pitches: np.ndarray


def _extract_note_intervals(track, ticks_per_beat, tempo):
    """(onset_sec, offset_sec, midi_pitch) tuples for a single string's MIDI track."""
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
                continue  # stray note_off with no matching note_on - skip defensively
            onset_sec = mido.tick2second(onset_tick, ticks_per_beat, tempo)
            offset_sec = mido.tick2second(abs_tick, ticks_per_beat, tempo)
            intervals.append((onset_sec, offset_sec, msg.note))
    return intervals


def _fit_score(pitches, open_midi):
    """Fraction of these pitches that land in a valid fret 0-19 window for this open string."""
    in_range = sum(1 for p in pitches if MIN_FRET <= (p - open_midi) <= MAX_FRET)
    return in_range / len(pitches)


# File order confirmed dataset-wide: tracks run high string -> low string.
# When a string is silent for a clip, EGDB's exporter sometimes omits its track
# rather than writing it empty - but it doesn't reorder the tracks that remain.
# So a present track's *position relative to the others* is reliable even when
# the absolute index isn't. Assignment is solved as an order-preserving
# alignment (present tracks -> an increasing subsequence of the 6 target
# strings) rather than free-for-all pitch matching, so a single coincidentally
# out-of-range note can't hijack the assignment the way it could with
# unconstrained highest-score-wins matching.
STRING_ORDER_HIGH_TO_LOW = ['e', 'B', 'G', 'D', 'A', 'E']


def _classify_tracks_to_strings(raw_by_track, string_midi, midi_path):
    """
    Aligns present (non-meta) tracks, in their file order, to an increasing
    subsequence of STRING_ORDER_HIGH_TO_LOW, maximizing total pitch-fit score.
    Returns {string_name: track_index_in_raw_by_track}; strings with no
    matched track (silent that clip) are simply absent from the result.
    """
    n_tracks = len(raw_by_track)
    n_strings = len(STRING_ORDER_HIGH_TO_LOW)

    def score(track_i, string_j):
        raw = raw_by_track[track_i]
        if not raw:
            return 0.0  # no notes = no evidence either way, still assignable
        pitches = [p for _, _, p in raw]
        return _fit_score(pitches, string_midi[STRING_ORDER_HIGH_TO_LOW[string_j]])

    NEG = float('-inf')
    # dp[i][j] = best total score assigning first i tracks into first j string-slots
    dp = [[NEG] * (n_strings + 1) for _ in range(n_tracks + 1)]
    take = [[False] * (n_strings + 1) for _ in range(n_tracks + 1)]
    for j in range(n_strings + 1):
        dp[0][j] = 0.0

    for i in range(1, n_tracks + 1):
        for j in range(i, n_strings + 1):
            skip_val = dp[i][j - 1]
            match_val = dp[i - 1][j - 1] + score(i - 1, j - 1) if dp[i - 1][j - 1] != NEG else NEG
            if match_val >= skip_val:
                dp[i][j], take[i][j] = match_val, True
            else:
                dp[i][j], take[i][j] = skip_val, False

    assigned_string = {}
    i, j = n_tracks, n_strings
    while i > 0:
        if take[i][j]:
            s = STRING_ORDER_HIGH_TO_LOW[j - 1]
            assigned_string[s] = i - 1
            sc = score(i - 1, j - 1)
            if raw_by_track[i - 1] and sc < CLASSIFICATION_CONFIDENCE_WARN:
                print(f"WARNING: {midi_path} track {i-1} matched to string '{s}' "
                      f"with only {sc:.0%} of its notes in a valid fret range - "
                      f"worth a manual look", flush=True)
            i -= 1
            j -= 1
        else:
            j -= 1

    return assigned_string


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
        print(f"WARNING: {midi_path} has {len(tempo_events)} tempo changes, "
              f"using only the first ({mido.tempo2bpm(tempo):.1f} BPM) - "
              f"timing may drift late in this clip", flush=True)

    non_meta_tracks = mid.tracks[1:]
    raw_by_track = [
        _extract_note_intervals(t, mid.ticks_per_beat, tempo) for t in non_meta_tracks
    ]
    assigned_string = _classify_tracks_to_strings(raw_by_track, string_midi, midi_path)

    notes_by_string = {}
    n_dropped = 0

    for s, open_midi in string_midi.items():
        if s not in assigned_string:
            notes_by_string[s] = None  # no track fit this string in this clip - genuinely silent
            continue

        raw = raw_by_track[assigned_string[s]]
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
            notes_by_string[s] = NoteData(
                intervals=np.array(list(zip(starts, ends))),
                pitches=np.array(pitches, dtype=float),
            )
        else:
            notes_by_string[s] = None  # build_label_array treats None as "no notes"

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