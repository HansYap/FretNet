import os
import mirdata
import librosa
import numpy as np
from config import STRING_MIDI
from labels import build_label_array, collapse_to_monophonic, stack_polyphonic_labels


def get_mirdata_dataset():
    data_home = os.path.expanduser('~/mir_datasets/guitarset')
    return mirdata.initialize('guitarset', data_home=data_home)

def get_solo_track_ids(mirdata_dataset):
    return [t for t in mirdata_dataset.track_ids if '_solo' in t]

def get_comp_track_ids(mirdata_dataset):
    return [t for t in mirdata_dataset.track_ids if '_comp' in t]

def get_all_track_ids(mirdata_dataset):
    # solo AND comp - comp is where the chords/strumming live
    return list(mirdata_dataset.track_ids)

def process_track(track_id, dataset, string_midi, sr=22050, hop_length=512):
    track = dataset.track(track_id)
    y, _ = librosa.load(track.audio_mic_path, sr=sr)
    cqt = librosa.cqt(y=y, sr=sr, hop_length=hop_length,
                       fmin=librosa.note_to_hz('C2'),
                       bins_per_octave=24, n_bins=96)

    cqt = librosa.amplitude_to_db(np.abs(cqt))

    label_arrays = {
        s: build_label_array(track.notes[s], open_midi, cqt.shape[1], sr, hop_length)
        for s, open_midi in string_midi.items()
    }

    poly_labels = stack_polyphonic_labels(label_arrays, string_midi)
    # cqt is the X input, poly_labels (6, n_frames) is the Y input
    return cqt, poly_labels


if __name__ == "__main__":
    mirdata_dataset = get_mirdata_dataset()
    all_ids = get_all_track_ids(mirdata_dataset)
    print(len(all_ids), "total tracks")

    cqt, poly_labels = process_track(all_ids[0], mirdata_dataset, STRING_MIDI)
    print("cqt shape:", cqt.shape)
    print("poly_labels shape:", poly_labels.shape)