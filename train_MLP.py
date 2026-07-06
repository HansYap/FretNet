import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import CRNN, MLPBaseline
from dataset_setup import get_train_val_datasets

def combined_loss(preds, targets):
    pitch_logits, string_logits, fret_logits = preds
    pitch_t, string_t, fret_t = targets
    return (F.cross_entropy(pitch_logits, pitch_t)
            + F.cross_entropy(string_logits, string_t)
            + F.cross_entropy(fret_logits, fret_t))

def run_epoch(model, loader, opt=None):
    is_train = opt is not None
    model.train() if is_train else model.eval()

    total_loss, total = 0.0, 0
    correct = {'pitch': 0, 'string': 0, 'fret': 0}

    with torch.set_grad_enabled(is_train):
        for x, string_t, fret_t, pitch_t in loader:
            string_t, fret_t, pitch_t = string_t.long(), fret_t.long(), pitch_t.long()

            if is_train:
                opt.zero_grad()

            preds = model(x.unsqueeze(1).float())
            loss = combined_loss(preds, (pitch_t, string_t, fret_t))

            if is_train:
                loss.backward()
                opt.step()

            # undo cross_entropy's batch averaging, weight by batch size
            total_loss += loss.item() * x.size(0)
            total += x.size(0)

            pitch_logits, string_logits, fret_logits = preds
            correct['pitch'] += (pitch_logits.argmax(dim=1) == pitch_t).sum().item()
            correct['string'] += (string_logits.argmax(dim=1) == string_t).sum().item()
            correct['fret']   += (fret_logits.argmax(dim=1)   == fret_t).sum().item()

    avg_loss = total_loss / total
    accuracy = {k: v / total for k, v in correct.items()}
    return avg_loss, accuracy


def train_model(model, train_ds, val_ds, epochs=10, batch_size=32, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        train_loss, _ = run_epoch(model, train_loader, opt)
        val_loss, val_acc = run_epoch(model, val_loader, opt=None)
        print(f"epoch {epoch}: train_loss={train_loss:.3f}  val_loss={val_loss:.3f}  "
              f"val_acc string/fret/pitch = {val_acc['string']:.1%}/{val_acc['fret']:.1%}/{val_acc['pitch']:.1%}")

    return val_acc


if __name__ == "__main__":
    train_ds, val_ds = get_train_val_datasets()
    print(f"train windows: {len(train_ds)}, val windows: {len(val_ds)}")

    print("\n===MLP baseline====")
    mlp = MLPBaseline()
    train_model(mlp, train_ds, val_ds, epochs=10)