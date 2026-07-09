import os
import glob
import numpy as np
from collections import defaultdict
from torch.utils.data import ConcatDataset
from dataset import GuitarSetDataset

CACHE_DIR = 'data/processed'


def load_cached_track(npz_path):
    data = np.load(npz_path)
    return data['cqt'], data['poly_labels']


def build_concat_dataset(npz_paths):
    datasets = [GuitarSetDataset(*load_cached_track(p)) for p in npz_paths]
    return ConcatDataset(datasets=datasets)


def get_player_id(npz_path):
    # GuitarSet filenames start with a 2-digit player number, e.g. "00_BN1-129-Eb_comp.npz"
    return os.path.basename(npz_path)[:2]


def get_train_val_datasets(val_player='05'):
    # split data based on guitar player
    npz_paths = sorted(glob.glob(os.path.join(CACHE_DIR, '*.npz')))
    assert len(npz_paths) > 0, "No cached tracks found, EMPTY"

    by_player = defaultdict(list)
    for p in npz_paths:
        by_player[get_player_id(p)].append(p)

    print("tracks per player:", {k: len(v) for k, v in sorted(by_player.items())})

    val_paths = by_player.pop(val_player)
    train_paths = [p for paths in by_player.values() for p in paths]

    print(f"train: {len(train_paths)} tracks, val (player {val_player}): {len(val_paths)} tracks")

    return build_concat_dataset(train_paths), build_concat_dataset(val_paths)