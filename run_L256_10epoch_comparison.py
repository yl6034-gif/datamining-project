import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

import json
from model_vit import VisionTransformer
from model_performer import PerformerViT
from train import run_experiment

# ============================================================
# L=256 Comparison: ViT vs Performer on CIFAR-10
# img_size=64, patch_size=4 -> sequence length L=(64/4)^2=256
# epochs=10 (reduced from 20 due to computational constraints)
# ============================================================

DATASET = 'cifar10'
NUM_CLASSES = 10
EPOCHS = 10
IMG_SIZE = 64  # L = (64/4)^2 = 256
model_vit = VisionTransformer(
    img_size=64,
    patch_size=4,
    num_classes=10
)

model_perf = PerformerViT(
    img_size=64,
    patch_size=4,
    num_classes=10,
    num_random_features=256
)
all_results = {}

# --- Model 1: ViT L=256 ---
print("\n" + "="*60)
print("Model 1: Regular ViT | CIFAR-10 | L=256 | 10 epochs")
print("="*60)
model_vit = VisionTransformer(
    img_size=IMG_SIZE,
    patch_size=4,
    num_classes=NUM_CLASSES
)
results_vit = run_experiment(
    model_vit, DATASET, 'ViT_L256',
    epochs=EPOCHS,
    img_size=64,  
    save_path='results_vit_cifar10_L256.json'
)
all_results['ViT_L256'] = results_vit

# --- Model 2: Performer L=256 ---
print("\n" + "="*60)
print("Model 2: Performer (FAVOR+, r=256) | CIFAR-10 | L=256 | 10 epochs")
print("="*60)
model_perf = PerformerViT(
    img_size=IMG_SIZE,
    patch_size=4,
    num_classes=NUM_CLASSES,
    num_random_features=256
)
results_perf = run_experiment(
    model_perf, DATASET, 'Performer_L256',
    epochs=EPOCHS,
    img_size=64,  
    use_redraw=True,
    save_path='results_performer_cifar10_L256.json'
)
all_results['Performer_L256'] = results_perf

# Save combined
with open('results_L256_comparison.json', 'w') as f:
    json.dump(all_results, f, indent=2)

# --- Summary ---
print(f"\n{'='*60}")
print("FINAL RESULTS - ViT vs Performer | CIFAR-10 | L=256 | 10 epochs")
print(f"{'='*60}")
print(f"{'Model':<20} {'Val Acc':>8} {'Train Time':>12} {'Inf Time':>10}")
print('-' * 55)
for name, r in all_results.items():
    print(f"{name:<20} {r['best_val_acc']:>8.4f} "
          f"{r['train_time']:>10.1f}s "
          f"{r['inf_time']:>8.2f}s")

vit_time = all_results['ViT_L256']['train_time']
perf_time = all_results['Performer_L256']['train_time']
print(f"\nSpeedup (ViT/Performer): {vit_time/perf_time:.2f}x")
print(f"Expected speedup at L=256: ~4x (O(N^2) vs O(N))")