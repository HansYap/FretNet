import os
import csv
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import CRNN
from dataset_setup import get_train_val_datasets
from config import NOT_PLAYED_CLASS

NOT_PLAYED = NOT_PLAYED_CLASS


def combined_loss(preds, targets):
    # preds: (batch, 6, 21), targets: (batch, 6)
    logits = preds.reshape(-1, preds.size(-1))  # (batch*6, 21)
    labels = targets.reshape(-1)                 # (batch*6,)
    return F.cross_entropy(logits, labels)


def run_epoch(model, loader, device, opt=None):
    is_train = opt is not None
    model.train() if is_train else model.eval()

    total_loss, total = 0.0, 0
    overall_correct, overall_total = 0, 0
    active_correct, active_total = 0, 0

    with torch.set_grad_enabled(is_train):
        for x, label_vec in loader:
            x = x.unsqueeze(1).float().to(device)
            label_vec = label_vec.long().to(device)  # (batch, 6)

            if is_train:
                opt.zero_grad()

            preds = model(x)  # (batch, 6, 21)
            loss = combined_loss(preds, label_vec)

            if is_train:
                loss.backward()
                opt.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            pred_classes = preds.argmax(dim=-1)       # (batch, 6)
            correct_mask = (pred_classes == label_vec)

            overall_correct += correct_mask.sum().item()
            overall_total += correct_mask.numel()

            # only count it where a string is actually being played
            active_mask = label_vec != NOT_PLAYED
            active_correct += (correct_mask & active_mask).sum().item()
            active_total += active_mask.sum().item()
    # two types of accuracy since a model logging all strings as not played in a imbalanced data instance can carry false negatives (actually played strings)
    avg_loss = total_loss / total
    overall_acc = overall_correct / overall_total
    active_acc = active_correct / active_total if active_total > 0 else 0.0
    return avg_loss, overall_acc, active_acc


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
        train_loss, train_overall_acc, train_active_acc = run_epoch(model, train_loader, device, opt)
        val_loss, val_overall_acc, val_active_acc = run_epoch(model, val_loader, device, opt=None)

        print(f"epoch {epoch}: train_loss={train_loss:.3f}  val_loss={val_loss:.3f}  "
              f"val_acc overall/active-string = {val_overall_acc:.1%}/{val_active_acc:.1%}",
              flush=True)

        log_rows.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_acc_overall": val_overall_acc,
            "val_acc_active_string": val_active_acc,
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
                "val_acc_overall": val_overall_acc,
                "val_acc_active_string": val_active_acc,
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