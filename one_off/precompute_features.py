import os 
import numpy as np
from config import STRING_MIDI
from audio_features import process_track, get_mirdata_dataset, get_solo_track_ids

CACHE_DIR = 'data/processed'

def precompute_all_solo_tracks():
    os.makedirs(CACHE_DIR, exist_ok=True)
    mirdata_dataset = get_mirdata_dataset()
    solo_ids = get_solo_track_ids(mirdata_dataset=mirdata_dataset)
    print(f"SOLOTRACKS===={len(solo_ids)}")

    for i, track_id in enumerate(solo_ids):
        out_path = os.path.join(CACHE_DIR, f'{track_id}.npz')
        if os.path.exists(out_path):
            continue

        cqt, labels = process_track(track_id, mirdata_dataset, STRING_MIDI)

        if len(labels['frame_idx']) == 0:
            print(f"[{i+1}/{len(solo_ids)}] {track_id}: 0 monophonic frames, SKIP")
            continue

        np.savez(
            out_path,
            cqt=cqt,
            frame_idx=labels['frame_idx'],
            string_idx=labels['string_idx'],
            fret=labels['fret'],
            pitch=labels['pitch'],
        )
        print(f"[{i+1}/{len(solo_ids)}] {track_id}: {len(labels['frame_idx'])} monophonic frames, CACHED")

if __name__ == "__main__":
    precompute_all_solo_tracks()