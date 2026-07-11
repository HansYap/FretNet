import os
import sys

# repo root isn't always on sys.path when this is run as `python one_off/precompute_features_egdb.py` -
# add it explicitly so the imports below work regardless of cwd/invocation method
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import STRING_MIDI
from egdb_features import process_egdb_track, get_egdb_clip_ids, TIMBRES

CACHE_DIR = 'data/processed'

# path to the EGDB folder after adding a Drive shortcut and mounting -
# adjust if your inspect_egdb.py run found a different path
EGDB_ROOT = '/content/drive/MyDrive/EGDB'


def precompute_all_egdb_tracks():
    os.makedirs(CACHE_DIR, exist_ok=True)
    clip_ids = get_egdb_clip_ids(EGDB_ROOT)
    total_jobs = len(clip_ids) * len(TIMBRES)
    print(f"EGDB clips: {len(clip_ids)}, timbres: {TIMBRES}", flush=True)
    print(f"total (clip, timbre) combinations to process: {total_jobs}", flush=True)

    n_written, n_skipped, total_dropped_notes = 0, 0, 0

    for i, clip_id in enumerate(clip_ids):
        for timbre in TIMBRES:
            # egdb_ prefix (not a 2-digit number) keeps these out of GuitarSet's
            # player-based train/val split logic - see dataset_setup.py
            out_path = os.path.join(CACHE_DIR, f'egdb_{clip_id:03d}_{timbre}.npz')
            if os.path.exists(out_path):
                n_skipped += 1
                continue

            cqt, poly_labels, n_dropped = process_egdb_track(
                clip_id, timbre, EGDB_ROOT, STRING_MIDI
            )
            total_dropped_notes += n_dropped

            np.savez(out_path, cqt=cqt, poly_labels=poly_labels)
            n_written += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(clip_ids):
            print(f"[{i+1}/{len(clip_ids)}] clips done "
                  f"({n_written} cached, {n_skipped} already existed)", flush=True)

    print(f"\nDONE. {n_written} new files cached, {n_skipped} skipped (already cached), "
          f"{total_dropped_notes} noisy notes dropped total "
          f"(pickup-bleed, outside valid fret 0-19)", flush=True)


if __name__ == "__main__":
    precompute_all_egdb_tracks()