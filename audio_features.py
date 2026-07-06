import os
import mirdata
import librosa
import numpy as np
from config import STRING_MIDI
from labels import build_label_array, collapse_to_monophonic


def get_mirdata_dataset():
    data_home = os.path.expanduser('~/mir_datasets/guitarset')
    return mirdata.initialize('guitarset', data_home=data_home)

def get_solo_track_ids(mirdata_dataset):
    return [t for t in mirdata_dataset.track_ids if '_solo' in t]

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

    labels = collapse_to_monophonic(label_arrays, string_midi)
    # cqt is the X input labels is the Y input
    return cqt, labels


if __name__ == "__main__":
    mirdata_dataset = get_mirdata_dataset()
    print(dir(mirdata_dataset))
    solo_ids = get_solo_track_ids(mirdata_dataset)
    print(len(solo_ids), "solo out of", len(mirdata_dataset.track_ids))

    # cqt, labels = process_track('05_Jazz3-137-Eb_solo', mirdata_dataset, STRING_MIDI)
    # print(len(labels['frame_idx']), "monophonic frames out of", cqt.shape[1])

    # i = np.where(labels['frame_idx'] == 32)[0][0]
    # print(labels['string_idx'][i], labels['fret'][i], labels['pitch'][i])



