import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
from dataset import GuitarSetDataset
from config import MIN_MIDI, N_PITCH_CLASSES

CACHE_DIR = 'data/processed'

def load_cached_track(npz_path):
    data = np.load(npz_path)
    labels = {
        'frame_idx': data['frame_idx'],
        'string_idx': data['string_idx'],
        'fret': data['fret'],
        'pitch': data['pitch'] - MIN_MIDI,
    }
    assert labels['pitch'].min() >= 0 and labels['pitch'].max() < N_PITCH_CLASSES, \
        f"pitch out of range in {npz_path}"
    
    return data['cqt'], labels


def build_concat_dataset(npz_paths):
    datasets = [GuitarSetDataset(*load_cached_track(p)) for p in npz_paths]
    return ConcatDataset(datasets=datasets)


def get_train_val_datasets(val_fraction=0.2, seed=2):
    npz_paths = sorted(glob.glob(os.path.join(CACHE_DIR, '*.npz')))
    assert len(npz_paths) > 0, "No cached tracks found, EMPTY"

    # shuffle here because CQT between tracks share zero overlap 
    train_paths, val_paths = train_test_split(npz_paths, test_size=val_fraction, random_state=seed)

    return build_concat_dataset(train_paths), build_concat_dataset(val_paths)
