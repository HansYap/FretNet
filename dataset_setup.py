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


def is_guitarset_track(npz_path):
    return get_player_id(npz_path).isdigit()


def get_train_val_datasets(val_player='05'):
    # split GuitarSet data based on guitar player; non-GuitarSet tracks (e.g. EGDB,
    # filenames like "egdb_001_DI.npz") have no player to hold out against, so they
    # always go to train 
    npz_paths = sorted(glob.glob(os.path.join(CACHE_DIR, '*.npz')))
    assert len(npz_paths) > 0, "No cached tracks found, EMPTY"

    by_player = defaultdict(list)
    extra_train_paths = []
    for p in npz_paths:
        if is_guitarset_track(p):
            by_player[get_player_id(p)].append(p)
        else:
            extra_train_paths.append(p)

    print("tracks per player:", {k: len(v) for k, v in sorted(by_player.items())})
    if extra_train_paths:
        print(f"non-GuitarSet tracks (always train): {len(extra_train_paths)}")

    val_paths = by_player.pop(val_player)
    train_paths = [p for paths in by_player.values() for p in paths] + extra_train_paths

    print(f"train: {len(train_paths)} tracks, val (player {val_player}): {len(val_paths)} tracks")

    return build_concat_dataset(train_paths), build_concat_dataset(val_paths)