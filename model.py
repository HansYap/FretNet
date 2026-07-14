import torch
import torch.nn as nn
from config import N_PITCH_CLASSES


class CRNN(nn.Module):
    def __init__(self, n_bins=96, n_strings=6, n_frets=20, dropout=0.3,
                 gru_hidden=128):
        super().__init__()
        # frets 0-19, plus one extra class for "not played" -> 21 classes per string
        self.n_classes_per_string = n_frets + 1

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # pool frequency only, keep time resolution
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),
        )
        # n_bins pooled twice (//2 then //2 again) = n_bins // 4
        self.rnn = nn.GRU(input_size=64 * (n_bins // 4), hidden_size=gru_hidden,
                           batch_first=True, bidirectional=True)
        # dropout on the GRU's output before it fans out into the six heads
        self.rnn_dropout = nn.Dropout(dropout)

        # one independent 21-way head per string (replaces string_head + fret_head + pitch_head)
        self.string_heads = nn.ModuleList([
            nn.Linear(gru_hidden * 2, self.n_classes_per_string) for _ in range(n_strings)
        ])

    # x: (batch, 1, n_bins, window)
    def forward(self, x):
        # (batch, C, F', window)
        x = self.conv(x)
        b, c, f, t = x.shape
        # (batch, time, features)
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)
        out, _ = self.rnn(x)

        center = t // 2
        # context from both directions
        center_out = out[:, center, :]
        center_out = self.rnn_dropout(center_out)

        # each head: (batch, 21) -> stacked: (batch, 6, 21)
        logits = torch.stack([head(center_out) for head in self.string_heads], dim=1)
        return logits


class MLPBaseline(nn.Module):
    def __init__(self, n_bins=96, window=9, n_strings=6, n_frets=20, n_pitch_classes=N_PITCH_CLASSES):
        super().__init__()
        input_size = n_bins * window
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
        )
        self.string_head = nn.Linear(128, n_strings)
        self.fret_head = nn.Linear(128, n_frets)
        self.pitch_head = nn.Linear(128, n_pitch_classes)

    # x: (batch, 1, n_bins, window)
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        h = self.net(x)
        return self.pitch_head(h), self.string_head(h), self.fret_head(h)