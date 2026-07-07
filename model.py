import torch
import torch.nn as nn
from config import N_PITCH_CLASSES

class CRNN(nn.Module):
    def __init__(self, n_bins=96, n_strings=6, n_frets=20, n_pitch_classes=N_PITCH_CLASSES,
                 dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # pool frequency only, keep time resolution
            nn.MaxPool2d((2, 1)),
            # zeros whole feature maps rather than individual pixels
            nn.Dropout2d(dropout),
        )
        self.rnn = nn.GRU(input_size=32 * (n_bins // 2), hidden_size=64, batch_first=True, bidirectional=True)
        # dropout on the GRU's output before it fans out into the three heads
        self.rnn_dropout = nn.Dropout(dropout)
        self.string_head = nn.Linear(128, n_strings)
        self.fret_head = nn.Linear(128, n_frets)
        self.pitch_head = nn.Linear(128, n_pitch_classes)
 
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
 
        return self.pitch_head(center_out), self.string_head(center_out), self.fret_head(center_out)



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