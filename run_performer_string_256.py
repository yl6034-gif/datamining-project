import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

import json
from model_performer_string import PerformerViTSTRING
from train import run_experiment

# ============================================================
# L=256 Experiment: Performer+Circulant-STRING on CIFAR-10
# img_size=64, patch_size=4 -> sequence length L=(64/4)^2=256
# epochs=10 (same as L256 comparison)
# All other settings identical to run_L256_10epoch_comparison.py
# ============================================================

DATASET     = 'cifar10'
NUM_CLASSES = 10
EPOCHS      = 10
IMG_SIZE    = 64  # L = (64/4)^2 = 256

print("\n" + "="*60)
print("Performer+Circulant-STRING | CIFAR-10 | L=256 | 10 epochs")
print("="*60)

model = PerformerViTSTRING(
    img_size=IMG_SIZE,
    patch_size=4,
    num_classes=NUM_CLASSES
)

results = run_experiment(
    model, DATASET, 'Performer_STRING_L256',
    epochs=EPOCHS,
    img_size=IMG_SIZE,
    use_redraw=False,
    save_path='results_string_cifar10_L256.json'
)

print(f"\n{'='*60}")
print("FINAL RESULTS - L=256 Full Comparison | CIFAR-10 | 10 epochs")
print(f"{'='*60}")
print(f"{'Model':<25} {'Val Acc':>8} {'Train Time':>12} {'Inf Time':>10}")
print('-' * 60)

# Load previous L256 results for comparison
vit_acc   = 0.7059; vit_time   = 1942.5;  vit_inf   = 11.92
perf_acc  = 0.6027; perf_time  = 17683.8; perf_inf  = 147.98
str_acc   = results['best_val_acc']
str_time  = results['train_time']
str_inf   = results['inf_time']

print(f"{'ViT_L256':<25} {vit_acc:>8.4f} {vit_time:>10.1f}s {vit_inf:>8.2f}s")
print(f"{'Performer_L256':<25} {perf_acc:>8.4f} {perf_time:>10.1f}s {perf_inf:>8.2f}s")
print(f"{'Performer+STRING_L256':<25} {str_acc:>8.4f} {str_time:>10.1f}s {str_inf:>8.2f}s")

print(f"\nSpeedup ViT/Performer:        {vit_time/perf_time:.2f}x")
print(f"Speedup ViT/STRING:           {vit_time/str_time:.2f}x")
print(f"Speedup Performer/STRING:     {perf_time/str_time:.2f}x")