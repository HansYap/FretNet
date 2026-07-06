import torch
from torch.utils.data import Dataset
import numpy as np

class GuitarSetDataset(Dataset):
    def __init__(self, cqt, labels, window=9):
        # unfiltered full cqt frames (how strong a frequency/pitch is at a give frame/time)
        self.cqt = cqt 
        self.frame_idx = labels['frame_idx']
        self.string_idx = labels['string_idx']
        self.fret = labels['fret']
        self.pitch = labels['pitch']
        self.half = window // 2

    def __len__(self):
        return len(self.frame_idx)

    def __getitem__(self, i):
        c = self.frame_idx[i]
        start, end = c - self.half, c + self.half + 1
        x = self.cqt[:, max(start, 0):end]
        # pad if window runs off either edge of the track
        pad_left = max(0, -start)
        pad_right = max(0, end - self.cqt.shape[1])
        if pad_left or pad_right:
            x = np.pad(x, ((0, 0), (pad_left, pad_right)))
        return x, self.string_idx[i], self.fret[i], self.pitch[i]

