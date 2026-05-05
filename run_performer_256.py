import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

import json
from model_performer import PerformerViT
from train import run_experiment

# ============================================================
# L=256: Performer (FAVOR+, r=128) on CIFAR-10
# img_size=64, patch_size=4 -> sequence length L=(64/4)^2=256
# epochs=10 (reduced due to computational constraints)
# ============================================================

DATASET     = 'cifar10'
NUM_CLASSES = 10
EPOCHS      = 10
IMG_SIZE    = 64

print("\n" + "="*60)
print("Performer (FAVOR+, r=128) | CIFAR-10 | L=256 | 10 epochs")
print("="*60)

model = PerformerViT(
    img_size=IMG_SIZE,
    patch_size=4,
    num_classes=NUM_CLASSES,
    num_random_features=128
)

results = run_experiment(
    model, DATASET, 'Performer_L256',
    epochs=EPOCHS,
    img_size=IMG_SIZE,
    use_redraw=True,
    save_path='results_performer_r128_cifar10_L256.json'
)

print(f"\nVal Acc:    {results['best_val_acc']:.4f}")
print(f"Train Time: {results['train_time']:.1f}s ({results['train_time']/60:.1f} min)")
print(f"Inf Time:   {results['inf_time']:.2f}s")