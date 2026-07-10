import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import CRNN
from dataset import GuitarSetDataset
from audio_features import process_track, get_mirdata_dataset
from config import STRING_MIDI, NOT_PLAYED_CLASS

NOT_PLAYED = NOT_PLAYED_CLASS

mirdata_dataset = get_mirdata_dataset()
cqt, poly_labels = process_track('05_Jazz3-137-Eb_comp', mirdata_dataset, STRING_MIDI)

assert poly_labels.shape[0] == 6, "expected one row per string"
assert poly_labels.min() >= 0 and poly_labels.max() <= NOT_PLAYED, "class out of range"

tiny_subset = GuitarSetDataset(cqt, poly_labels)


def combined_loss(preds, targets):
    logits = preds.reshape(-1, preds.size(-1))
    labels = targets.reshape(-1)
    return F.cross_entropy(logits, labels)


model = CRNN()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

loader = DataLoader(tiny_subset, batch_size=8, shuffle=True)


def train_run(model, opt, loader):
    for step, (x, label_vec) in enumerate(loader):
        opt.zero_grad()
        preds = model(x.unsqueeze(1).float())
        loss = combined_loss(preds, label_vec.long())
        loss.backward()
        opt.step()

        pred_classes = preds.argmax(dim=-1)
        active_mask = label_vec != NOT_PLAYED
        active_acc = (pred_classes == label_vec)[active_mask].float().mean().item() if active_mask.any() else float('nan')

        print(step, loss.item(), "active-string acc:", active_acc)
        if step > 20:
            break


train_run(model, opt, loader)