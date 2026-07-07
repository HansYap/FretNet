import os
import csv
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import CRNN
from dataset_setup import get_train_val_datasets


def combined_loss(preds, targets):
    pitch_logits, string_logits, fret_logits = preds
    pitch_t, string_t, fret_t = targets
    return (F.cross_entropy(pitch_logits, pitch_t)
            + F.cross_entropy(string_logits, string_t)
            + F.cross_entropy(fret_logits, fret_t))


def run_epoch(model, loader, device, opt=None):
    is_train = opt is not None
    model.train() if is_train else model.eval()

    total_loss, total = 0.0, 0
    correct = {'pitch': 0, 'string': 0, 'fret': 0}

    with torch.set_grad_enabled(is_train):
        for x, string_t, fret_t, pitch_t in loader:
            x = x.unsqueeze(1).float().to(device)
            string_t = string_t.long().to(device)
            fret_t = fret_t.long().to(device)
            pitch_t = pitch_t.long().to(device)

            if is_train:
                opt.zero_grad()

            preds = model(x)
            loss = combined_loss(preds, (pitch_t, string_t, fret_t))

            if is_train:
                loss.backward()
                opt.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            pitch_logits, string_logits, fret_logits = preds
            correct['pitch'] += (pitch_logits.argmax(dim=1) == pitch_t).sum().item()
            correct['string'] += (string_logits.argmax(dim=1) == string_t).sum().item()
            correct['fret']   += (fret_logits.argmax(dim=1)   == fret_t).sum().item()

    avg_loss = total_loss / total
    accuracy = {k: v / total for k, v in correct.items()}
    return avg_loss, accuracy


def train_crnn(checkpoint_dir, epochs=50, batch_size=32, lr=1e-3,
                patience=5, num_workers=2, weight_decay=1e-3, dropout=0.3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    train_ds, val_ds = get_train_val_datasets()
    print(f"train windows: {len(train_ds)}, val windows: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    model = CRNN(dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    os.makedirs(checkpoint_dir, exist_ok=True)
    log_path = os.path.join(checkpoint_dir, "training_log.csv")
    log_rows = []

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss, _ = run_epoch(model, train_loader, device, opt)
        val_loss, val_acc = run_epoch(model, val_loader, device, opt=None)

        print(f"epoch {epoch}: train_loss={train_loss:.3f}  val_loss={val_loss:.3f}  "
              f"val_acc string/fret/pitch = {val_acc['string']:.1%}/{val_acc['fret']:.1%}/{val_acc['pitch']:.1%}",
              flush=True)

        log_rows.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_acc_string": val_acc['string'],
            "val_acc_fret": val_acc['fret'],
            "val_acc_pitch": val_acc['pitch'],
        })
        # rewrite the whole CSV each epoch in case disconnected from colab
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            ckpt_path = os.path.join(checkpoint_dir, "crnn_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_state,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"NEW BEST=== saved to {ckpt_path}", flush=True)
        else:
            epochs_no_improve += 1
            print(f"NO IMPROVEMENT ===== for {epochs_no_improve} epoch(s)", flush=True)

        if epochs_no_improve >= patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


if __name__ == "__main__":
    train_crnn(checkpoint_dir="/content/drive/MyDrive/fretnet/checkpoints")