import os
import numpy as np
from config import STRING_MIDI
from audio_features import process_track, get_mirdata_dataset, get_all_track_ids

CACHE_DIR = 'data/processed'

def precompute_all_tracks():
    os.makedirs(CACHE_DIR, exist_ok=True)
    mirdata_dataset = get_mirdata_dataset()
    all_ids = get_all_track_ids(mirdata_dataset=mirdata_dataset)
    print(f"ALL TRACKS ==== {len(all_ids)}")

    for i, track_id in enumerate(all_ids):
        out_path = os.path.join(CACHE_DIR, f'{track_id}.npz')
        if os.path.exists(out_path):
            continue

        cqt, poly_labels = process_track(track_id, mirdata_dataset, STRING_MIDI)

        np.savez(
            out_path,
            cqt=cqt,
            poly_labels=poly_labels,
        )
        print(f"[{i+1}/{len(all_ids)}] {track_id}: {cqt.shape[1]} frames, CACHED")

if __name__ == "__main__":
    precompute_all_tracks()