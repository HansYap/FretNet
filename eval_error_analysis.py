import torch
from torch.utils.data import DataLoader

from model import CRNN
from dataset_setup import get_train_val_datasets
from config import NOT_PLAYED_CLASS, STRING_MIDI

NOT_PLAYED = NOT_PLAYED_CLASS
STRINGS_ORDER = list(STRING_MIDI.keys())  # E, A, D, G, B, e - matches labels.py / model head order
STRING_MIDI_LIST = [STRING_MIDI[s] for s in STRINGS_ORDER]
CHECKPOINT_PATH = "/content/drive/MyDrive/FretNet/checkpoints/crnn_best.pt"


def load_model(device):
    model = CRNN().to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.3f}", flush=True)
    return model


def collect_predictions(model, val_loader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, label_vec in val_loader:
            x = x.unsqueeze(1).float().to(device)
            preds = model(x).argmax(dim=-1).cpu()
            all_preds.append(preds)
            all_labels.append(label_vec.long())
    return torch.cat(all_preds), torch.cat(all_labels)  # (N, 6) each


def neighbors_of(s_idx):
    ns = []
    if s_idx - 1 >= 0:
        ns.append(s_idx - 1)
    if s_idx + 1 < 6:
        ns.append(s_idx + 1)
    return ns


def bucket_distance(d):
    if d == 1:
        return '1'
    if d == 2:
        return '2'
    if d == 3:
        return '3'
    if 4 <= d <= 6:
        return '4-6'
    return '7+'


def evaluate_error_structure():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    _, val_ds = get_train_val_datasets()
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    model = load_model(device)
    preds, labels = collect_predictions(model, val_loader, device)

    print("\n" + "=" * 78)
    print("PER-STRING ERROR BREAKDOWN (GuitarSet player-05 val set)")
    print("=" * 78)

    total_wrong_all = 0
    total_explained_all = 0

    for s_idx, s_name in enumerate(STRINGS_ORDER):
        true_s = labels[:, s_idx]
        pred_s = preds[:, s_idx]

        active_mask = true_s != NOT_PLAYED
        active_total = active_mask.sum().item()
        wrong_mask = active_mask & (pred_s != true_s)
        wrong_total = wrong_mask.sum().item()
        correct = active_total - wrong_total
        recall = correct / active_total

        miss_mask = wrong_mask & (pred_s == NOT_PLAYED)
        miss_count = miss_mask.sum().item()

        wrong_fret_mask = wrong_mask & (pred_s != NOT_PLAYED)
        wrong_fret_count = wrong_fret_mask.sum().item()

        dist = (pred_s[wrong_fret_mask] - true_s[wrong_fret_mask]).abs()
        buckets = {'1': 0, '2': 0, '3': 0, '4-6': 0, '7+': 0}
        for d in dist.tolist():
            buckets[bucket_distance(d)] += 1

        # cross-string same-pitch confusion: for each wrong prediction on string s, does a
        # neighboring string's head predict the exact true pitch, while that neighbor's own
        # prediction is itself wrong relative to ITS OWN ground truth (i.e. not just a genuine
        # coincident note the neighbor was right to predict)?
        true_pitch_s = STRING_MIDI_LIST[s_idx] + true_s
        explained_mask = torch.zeros_like(wrong_mask)
        for n_idx in neighbors_of(s_idx):
            pred_n = preds[:, n_idx]
            true_n = labels[:, n_idx]
            n_active_pred = pred_n != NOT_PLAYED
            pred_pitch_n = STRING_MIDI_LIST[n_idx] + pred_n
            same_pitch = n_active_pred & (pred_pitch_n == true_pitch_s)
            neighbor_spurious = pred_n != true_n
            explained_mask |= (wrong_mask & same_pitch & neighbor_spurious)

        explained_count = explained_mask.sum().item()

        print(f"\n{s_name} string: active_total={active_total}, recall={recall:.1%}")
        print(f"  miss (predicted not-played): {miss_count} ({miss_count / active_total:.1%} of active)")
        print(f"  wrong-fret (predicted a different real fret): {wrong_fret_count} "
              f"({wrong_fret_count / active_total:.1%} of active)")
        print(f"    fret-error distance among those: 1={buckets['1']} 2={buckets['2']} "
              f"3={buckets['3']} 4-6={buckets['4-6']} 7+={buckets['7+']}")
        print(f"  cross-string same-pitch explained: {explained_count}/{wrong_total} wrong predictions "
              f"({explained_count / wrong_total:.1%}) - a neighboring string's head fired at the exact "
              f"true pitch while this string missed it")

        total_wrong_all += wrong_total
        total_explained_all += explained_count

    print("\n" + "=" * 78)
    print(f"OVERALL: {total_explained_all}/{total_wrong_all} wrong predictions "
          f"({total_explained_all / total_wrong_all:.1%}) explained by cross-string same-pitch confusion")
    print("=" * 78)

if __name__ == "__main__":
    evaluate_error_structure()