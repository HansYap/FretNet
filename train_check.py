import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import CRNN, MLPBaseline
from dataset import GuitarSetDataset 
from audio_features import process_track, get_mirdata_dataset
from config import STRING_MIDI, MIN_MIDI, N_PITCH_CLASSES

mirdata_dataset = get_mirdata_dataset()
cqt, labels = process_track('05_Jazz3-137-Eb_solo', mirdata_dataset, STRING_MIDI)

labels['pitch'] = labels['pitch'] - MIN_MIDI

assert labels['pitch'].min() >= 0 and labels['pitch'].max() < N_PITCH_CLASSES, "pitch out of range"
assert labels['fret'].min() >= 0 and labels['fret'].max() < 20, "fret out of range"
assert labels['string_idx'].min() >= 0 and labels['string_idx'].max() < 6, "string out of range"

tiny_subset = GuitarSetDataset(cqt, labels)

def combined_loss(preds, targets):
    pitch_logits, string_logits, fret_logits = preds
    pitch_t, string_t, fret_t = targets
    return (F.cross_entropy(pitch_logits, pitch_t)
            + F.cross_entropy(string_logits, string_t)
            + F.cross_entropy(fret_logits, fret_t))

model = CRNN()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
model_Baseline = MLPBaseline()
opt_Baseline = torch.optim.Adam(model_Baseline.parameters(), lr=1e-3)

loader = DataLoader(tiny_subset, batch_size=8, shuffle=True)

def train_run(model, opt, loader):
    for step, (x, string_t, fret_t, pitch_t) in enumerate(loader):
        opt.zero_grad()
        preds = model(x.unsqueeze(1).float())
        loss = combined_loss(preds, (pitch_t.long(), string_t.long(), fret_t.long()))
        loss.backward()
        opt.step()
        print(step, loss.item())
        if step > 20:
            break

train_run(model, opt, loader)
train_run(model_Baseline, opt_Baseline, loader)