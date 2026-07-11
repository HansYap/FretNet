import mido
from config import STRING_MIDI
from egdb_features import load_egdb_notes

TICKS_PER_BEAT = 480
TEMPO = mido.bpm2tempo(120)  


def make_track(events):
    track = mido.MidiTrack()
    events = sorted(events, key=lambda e: e[0])
    prev_tick = 0
    for abs_tick, msg in events:
        msg.time = abs_tick - prev_tick
        track.append(msg)
        prev_tick = abs_tick
    track.append(mido.MetaMessage('end_of_track', time=0))
    return track


def build_synthetic_midi(path):
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)

    # track 0: meta/tempo, no notes 
    meta_track = make_track([
        (0, mido.MetaMessage('set_tempo', tempo=TEMPO)),
    ])
    mid.tracks.append(meta_track)

    # track 1 (high e, open=64): one note at fret 5 -> pitch 69, from tick 0 to 480 (1 beat = 0.5s @ 120bpm)
    mid.tracks.append(make_track([
        (0, mido.Message('note_on', note=69, velocity=90)),
        (480, mido.Message('note_off', note=69, velocity=0)),
    ]))

    # track 2 (B, open=59): one note at fret 2 -> pitch 61, from tick 480 to 960
    mid.tracks.append(make_track([
        (480, mido.Message('note_on', note=61, velocity=90)),
        (960, mido.Message('note_off', note=61, velocity=0)),
    ]))

    # track 3 (G, open=55): deliberately out-of-range note (pitch 50, implied fret -5)
    # to check it gets dropped, not clamped
    mid.tracks.append(make_track([
        (0, mido.Message('note_on', note=50, velocity=90)),
        (240, mido.Message('note_off', note=50, velocity=0)),
    ]))

    # track 4 (D, open=50): empty (no notes) - checks the None branch
    mid.tracks.append(make_track([]))

    # track 5 (A, open=45): open string note -> fret 0, pitch 45
    mid.tracks.append(make_track([
        (0, mido.Message('note_on', note=45, velocity=90)),
        (960, mido.Message('note_off', note=45, velocity=0)),
    ]))

    # track 6 (low E, open=40): fret 19 (max valid) -> pitch 59
    mid.tracks.append(make_track([
        (0, mido.Message('note_on', note=59, velocity=90)),
        (480, mido.Message('note_off', note=59, velocity=0)),
    ]))

    mid.save(path)


def run_checks():
    path = '/tmp/synthetic_egdb_clip.mid'
    build_synthetic_midi(path)
    notes_by_string, n_dropped = load_egdb_notes(path, STRING_MIDI)

    checks = []

    # e string: fret 5, onset 0.0s, offset 0.5s (480 ticks @ 480 tpb, 120bpm = 1 beat = 0.5s)
    e = notes_by_string['e']
    checks.append(("e string exists", e is not None))
    checks.append(("e string pitch", e.pitches[0] == 69))
    checks.append(("e string onset ~0.0s", abs(e.intervals[0, 0] - 0.0) < 1e-6))
    checks.append(("e string offset ~0.5s", abs(e.intervals[0, 1] - 0.5) < 1e-6))

    # B string: fret 2, onset 0.5s, offset 1.0s
    b = notes_by_string['B']
    checks.append(("B string onset ~0.5s", abs(b.intervals[0, 0] - 0.5) < 1e-6))
    checks.append(("B string offset ~1.0s", abs(b.intervals[0, 1] - 1.0) < 1e-6))

    # G string: out-of-range note should be dropped entirely
    checks.append(("G string dropped (None)", notes_by_string['G'] is None))
    checks.append(("n_dropped == 1", n_dropped == 1))

    # D string: no notes at all -> None
    checks.append(("D string is None (no notes)", notes_by_string['D'] is None))

    # A string: open note, fret 0
    a = notes_by_string['A']
    checks.append(("A string pitch (open)", a.pitches[0] == 45))

    # low E string: fret 19 (boundary, should NOT be dropped)
    low_e = notes_by_string['E']
    checks.append(("low E fret 19 kept", low_e is not None and low_e.pitches[0] == 59))

    print(f"{'PASS' if all(ok for _, ok in checks) else 'FAIL - see below'}")
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")


if __name__ == "__main__":
    run_checks()