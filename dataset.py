import torch
from torch.utils.data import Dataset
import numpy as np

class GuitarSetDataset(Dataset):
    def __init__(self, cqt, poly_labels, window=9):
        # unfiltered full cqt frames (how strong a frequency/pitch is at a given frame/time)
        self.cqt = cqt
        # (6, n_frames), one class (0-20) per string, per frame
        self.poly_labels = poly_labels
        self.half = window // 2

    def __len__(self):
        return self.cqt.shape[1]

    def __getitem__(self, i):
        start, end = i - self.half, i + self.half + 1
        x = self.cqt[:, max(start, 0):end]
        # pad if window runs off either edge of the track
        pad_left = max(0, -start)
        pad_right = max(0, end - self.cqt.shape[1])
        if pad_left or pad_right:
            x = np.pad(x, ((0, 0), (pad_left, pad_right)))
        # (6,) - one class index (0-20) per string, for this center frame
        label_vec = self.poly_labels[:, i]
        return x, label_vec