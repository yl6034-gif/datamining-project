import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

import json
from model_performer import PerformerViT
from train import run_experiment

datasets = {
    'mnist':         10,
    'fashion_mnist': 10,
    'cifar10':       10,
    'cifar100':      100,
}

all_results = {}

for dataset_name, num_classes in datasets.items():
    print(f"\n{'='*50}")
    print(f"Running Performer (FAVOR+, r=256) on {dataset_name}")
    print(f"{'='*50}")

    model = PerformerViT(img_size=32, num_classes=num_classes,
                         num_random_features=256)
    results = run_experiment(
        model, dataset_name, 'Performer',
        epochs=20, use_redraw=True,
        save_path=f'results_performer_{dataset_name}.json'
    )
    all_results[dataset_name] = results

# Save all results
with open('results_performer_all.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*50}")
print("FINAL RESULTS - Performer (FAVOR+, r=256)")
print(f"{'='*50}")
print(f"{'Dataset':<15} {'Val Acc':>8} {'Train Time':>12} {'Inf Time':>10}")
print('-' * 50)
for name, r in all_results.items():
    print(f"{name:<15} {r['best_val_acc']:>8.4f} "
          f"{r['train_time']:>10.1f}s "
          f"{r['inf_time']:>8.2f}s")