import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_loader import get_dataset


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def redraw_random_features(model):
    """Redraw random features for all FAVOR+ attention modules."""
    for module in model.modules():
        if hasattr(module, 'random_matrix') and hasattr(module, '_sample_ortho_features'):
            module.random_matrix = module._sample_ortho_features().to(
                module.random_matrix.device
            )


def train_one_epoch(model, loader, optimizer, loss_fn, device, use_redraw=False):
    model.train()
    correct = total = 0
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    # Redraw random features after each epoch (Performer only)
    if use_redraw:
        redraw_random_features(model)

    return total_loss / len(loader), correct / total


def evaluate(model, loader, loss_fn, device):
    model.eval()
    correct = total = 0
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    return total_loss / len(loader), correct / total

def run_experiment(model, dataset_name, model_name,
                   epochs=20, batch_size=128, lr=3e-4,
                   img_size=32, use_redraw=False, save_path=None):

    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Model: {model_name} | Dataset: {dataset_name} | Epochs: {epochs}")

    train_loader, val_loader, _ = get_dataset(dataset_name,
                                           img_size=img_size,
                                           batch_size=batch_size)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    train_start = time.time()
    val_acc_curve   = []
    train_acc_curve = []
    val_loss_curve  = []
    train_loss_curve = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, use_redraw
        )
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        # Save curves
        val_acc_curve.append(round(val_acc, 4))
        train_acc_curve.append(round(train_acc, 4))
        val_loss_curve.append(round(val_loss, 4))
        train_loss_curve.append(round(train_loss, 4))

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    total_train_time = time.time() - train_start

    # Measure inference time
    inf_start = time.time()
    evaluate(model, val_loader, loss_fn, device)
    inf_time = time.time() - inf_start

    print(f"\n--- Results ---")
    print(f"Best val acc:     {best_val_acc:.4f}")
    print(f"Total train time: {total_train_time:.1f}s")
    print(f"Inference time:   {inf_time:.2f}s")

    results = {
        'model':            model_name,
        'dataset':          dataset_name,
        'best_val_acc':     round(best_val_acc, 4),
        'train_time':       round(total_train_time, 1),
        'inf_time':         round(inf_time, 2),
        'val_acc_curve':    val_acc_curve,
        'train_acc_curve':  train_acc_curve,
        'val_loss_curve':   val_loss_curve,
        'train_loss_curve': train_loss_curve,
    }

    # Auto save after each dataset
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {save_path}")

    return results


if __name__ == '__main__':
    from model_vit import VisionTransformer

    model = VisionTransformer(num_classes=10)
    results = run_experiment(
        model, 'mnist', 'ViT', epochs=3
    )

