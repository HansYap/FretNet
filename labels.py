import librosa
import numpy as np

def build_label_array(notes, open_string_midi, n_frames, sr, hop_length):

    label_array = np.full(n_frames, -1)

    if notes is None:
        return label_array

    # (sr / hop_length) × time to claculate frames for each note
    # NOTE == librosa rounds down (floor) not up but for end frame have to round UP
    timeframe_start = librosa.time_to_frames(notes.intervals[:, 0], sr=sr, hop_length=hop_length)

    timeframe_end = (notes.intervals[:, 1] * sr) / hop_length
    timeframe_end = np.ceil(timeframe_end).astype(int)

    frets = np.round(notes.pitches).astype(int) - open_string_midi

    for start, end, fret in zip(timeframe_start, timeframe_end, frets):
        label_array[start:end] = fret   

    return label_array


def collapse_to_monophonic(label_arrays, string_midi):
    strings_order = list(string_midi.keys()) 
    stacked = np.stack([label_arrays[s] for s in strings_order])

    # only look for single string frame
    is_active = stacked != -1                     
    active_count = is_active.sum(axis=0)           
    mono_mask = active_count == 1                   

    string_idx = np.argmax(is_active, axis=0)       
    frame_range = np.arange(stacked.shape[1])
    fret_per_frame = stacked[string_idx, frame_range]  

    open_midi_lookup = np.array([string_midi[s] for s in strings_order])
    pitch_per_frame = fret_per_frame + open_midi_lookup[string_idx]

    # keep only monophonic frames
    valid_frames = np.where(mono_mask)[0]
    # return 1D frames of active frames and corresponding string played and which fret, 
    return {
        'frame_idx': valid_frames,
        'string_idx': string_idx[valid_frames],
        'fret': fret_per_frame[valid_frames],
        'pitch': pitch_per_frame[valid_frames],
    }