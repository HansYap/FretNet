"""
Builds synthetic MIDI files matching EGDB's real structure with known notes on
known strings, then checks load_egdb_notes() recovers the correct frets and
timings - including the real-data bug where a silent string's track is
omitted entirely rather than written empty, which shifts every later track's
position and breaks any fixed track_idx -> string mapping.
Run this before trusting the adapter against real EGDB data.
"""
import mido
from config import STRING_MIDI
from egdb_features import load_egdb_notes

TICKS_PER_BEAT = 480
TEMPO = mido.bpm2tempo(120)  # 500000 us/quarter
STRING_ORDER_HIGH_TO_LOW = ['e', 'B', 'G', 'D', 'A', 'E']


def make_track(events):
    """events: list of (abs_tick, msg) -> converts to delta-time track"""
    track = mido.MidiTrack()
    events = sorted(events, key=lambda e: e[0])
    prev_tick = 0
    for abs_tick, msg in events:
        msg.time = abs_tick - prev_tick
        track.append(msg)
        prev_tick = abs_tick
    track.append(mido.MetaMessage('end_of_track', time=0))
    return track


def _note(pitch, start_tick, end_tick):
    return [
        (start_tick, mido.Message('note_on', note=pitch, velocity=90)),
        (end_tick, mido.Message('note_off', note=pitch, velocity=0)),
    ]


def build_bad_note_midi(path):
    """All 6 string tracks physically present. G has one deliberately
    out-of-range note (tests drop-not-clamp). Low E sits at the fret-19
    boundary (must be kept, not treated as out-of-range)."""
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    mid.tracks.append(make_track([(0, mido.MetaMessage('set_tempo', tempo=TEMPO))]))
    mid.tracks.append(make_track(_note(69, 0, 480)))     # e: fret 5
    mid.tracks.append(make_track(_note(61, 480, 960)))   # B: fret 2
    mid.tracks.append(make_track(_note(50, 0, 240)))     # G: fret -5, invalid
    mid.tracks.append(make_track([]))                    # D: silent, written empty
    mid.tracks.append(make_track(_note(45, 0, 960)))     # A: fret 0 (open)
    mid.tracks.append(make_track(_note(59, 0, 480)))     # low E: fret 19
    mid.save(path)


def build_missing_track_midi(path, strings_present):
    """Only writes tracks for strings_present (subset of STRING_ORDER_HIGH_TO_LOW,
    in that order) - the others are omitted from the file entirely, mimicking
    the real EGDB bug. Each present string gets 3 notes spanning frets +2, +10,
    +17 relative to its open pitch - a WIDE spread, not clustered near the open
    string. Adjacent strings' valid-fret windows overlap so heavily (20-semitone
    window, ~5-semitone string spacing) that a narrow low-fret cluster is
    mathematically compatible with several neighboring strings at an identical
    score - not a realistic clip, and not something any algorithm could resolve
    from pitch content alone. A wide spread breaks that tie the way real note
    diversity across a ~30s clip would."""
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    mid.tracks.append(make_track([(0, mido.MetaMessage('set_tempo', tempo=TEMPO))]))
    for s in strings_present:
        open_pitch = STRING_MIDI[s]
        events = (_note(open_pitch + 2, 0, 240)
                  + _note(open_pitch + 10, 240, 480)
                  + _note(open_pitch + 17, 480, 720))
        mid.tracks.append(make_track(events))
    mid.save(path)


def check_bad_note_scenario():
    path = '/tmp/synthetic_bad_note.mid'
    build_bad_note_midi(path)
    notes_by_string, n_dropped = load_egdb_notes(path, STRING_MIDI)

    checks = [
        ("e string pitch (fret 5)", notes_by_string['e'] is not None and notes_by_string['e'].pitches[0] == 69),
        ("B string pitch (fret 2)", notes_by_string['B'] is not None and notes_by_string['B'].pitches[0] == 61),
        ("G string dropped (out-of-range note)", notes_by_string['G'] is None),
        ("n_dropped == 1", n_dropped == 1),
        ("D string is None (written empty)", notes_by_string['D'] is None),
        ("A string pitch (open, fret 0)", notes_by_string['A'] is not None and notes_by_string['A'].pitches[0] == 45),
        ("low E fret 19 kept (boundary)", notes_by_string['E'] is not None and notes_by_string['E'].pitches[0] == 59),
    ]
    return checks


def check_missing_track_scenario(label, strings_present):
    path = f'/tmp/synthetic_missing_{label}.mid'
    build_missing_track_midi(path, strings_present)
    notes_by_string, n_dropped = load_egdb_notes(path, STRING_MIDI)

    checks = [(f"n_dropped == 0 ({label})", n_dropped == 0)]
    for s in STRING_ORDER_HIGH_TO_LOW:
        if s in strings_present:
            expected_pitch = STRING_MIDI[s] + 2  # first note written is the fret+2 one
            got = notes_by_string[s]
            checks.append((
                f"'{s}' present & correctly identified ({label})",
                got is not None and got.pitches[0] == expected_pitch
            ))
        else:
            checks.append((
                f"'{s}' correctly absent/silent ({label})",
                notes_by_string[s] is None
            ))
    return checks


def run_checks():
    all_scenarios = [
        ("bad_note (all 6 tracks present)", check_bad_note_scenario()),
        ("missing middle (D omitted)", check_missing_track_scenario(
            "missing_middle", ['e', 'B', 'G', 'A', 'E'])),
        ("missing first (e omitted)", check_missing_track_scenario(
            "missing_first", ['B', 'G', 'D', 'A', 'E'])),
        ("missing last (low E omitted)", check_missing_track_scenario(
            "missing_last", ['e', 'B', 'G', 'D', 'A'])),
        ("multiple missing, non-adjacent (only e, D, low E present)", check_missing_track_scenario(
            "multiple_missing", ['e', 'D', 'E'])),
        ("only one track present (G only)", check_missing_track_scenario(
            "single_track", ['G'])),
    ]

    all_ok = True
    for label, checks in all_scenarios:
        scenario_ok = all(ok for _, ok in checks)
        all_ok &= scenario_ok
        print(f"\n=== {label}: {'PASS' if scenario_ok else 'FAIL'} ===")
        for name, ok in checks:
            print(f"  [{'OK' if ok else 'FAIL'}] {name}")

    print(f"\n{'ALL SCENARIOS PASS' if all_ok else 'SOME SCENARIOS FAILED - see above'}")
    return all_ok


if __name__ == "__main__":
    ok = run_checks()
    import sys
    sys.exit(0 if ok else 1)