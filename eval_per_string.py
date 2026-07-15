import torch
from torch.utils.data import DataLoader

from model import CRNN
from dataset_setup import get_train_val_datasets
from config import NOT_PLAYED_CLASS, STRING_MIDI

NOT_PLAYED = NOT_PLAYED_CLASS
STRINGS_ORDER = list(STRING_MIDI.keys())  # E, A, D, G, B, e - matches labels.py / model head order
CHECKPOINT_PATH = "/content/drive/MyDrive/FretNet/checkpoints/crnn_best.pt"


def load_model(device):
    model = CRNN().to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.3f}, "
          f"val_acc active-string={ckpt['val_acc_active_string']:.1%}", flush=True)
    return model


def collect_predictions(model, val_loader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, label_vec in val_loader:
            x = x.unsqueeze(1).float().to(device)
            preds = model(x).argmax(dim=-1).cpu()  # (batch, 6)
            all_preds.append(preds)
            all_labels.append(label_vec.long())
    return torch.cat(all_preds), torch.cat(all_labels)  # (N, 6) each


def per_string_recall(preds, labels):
    print("\nPer-string active recall (GuitarSet player-05 val set):")
    for s_idx, s_name in enumerate(STRINGS_ORDER):
        true_s = labels[:, s_idx]
        pred_s = preds[:, s_idx]
        active_mask = true_s != NOT_PLAYED
        active_total = active_mask.sum().item()
        correct = ((pred_s == true_s) & active_mask).sum().item()
        print(f"   {s_name}: {correct / active_total:.1%}  ({correct}/{active_total})")


def off_by_one_rate(preds, labels):
    # numerator: wrong-fret errors (both pred and true are real frets) off by exactly 1 fret
    # denominator: ALL wrong active predictions (misses + wrong-fret combined), summed across strings
    total_wrong = 0
    off_by_one = 0
    for s_idx in range(6):
        true_s = labels[:, s_idx]
        pred_s = preds[:, s_idx]
        active_mask = true_s != NOT_PLAYED
        wrong_mask = active_mask & (pred_s != true_s)
        total_wrong += wrong_mask.sum().item()

        wrong_fret_mask = wrong_mask & (pred_s != NOT_PLAYED)
        dist = (pred_s[wrong_fret_mask] - true_s[wrong_fret_mask]).abs()
        off_by_one += (dist == 1).sum().item()

    rate = off_by_one / total_wrong
    print(f"\noff-by-one-fret rate among wrong active predictions: {rate:.1%}")


def evaluate_per_string():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    _, val_ds = get_train_val_datasets()
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    model = load_model(device)
    preds, labels = collect_predictions(model, val_loader, device)

    per_string_recall(preds, labels)
    off_by_one_rate(preds, labels)

if __name__ == "__main__":
    evaluate_per_string()