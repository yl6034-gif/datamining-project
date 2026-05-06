import os
import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. Load JSON results
# =========================
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "result"

def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)

vit = load_json("results_vit_all.json")
perf = load_json("results_performer_all.json")
string = load_json("results_string_all.json")

# L=256 experiments
# L=256 experiments
vit_256       = load_json("results_vit_cifar10_L256.json")
performer_256 = load_json("results_performer_cifar10_L256.json")
string_256    = load_json("results_string_cifar10_L256.json")


# =========================
# 2. Main L=64 learning curves
# =========================

def plot_l64_learning_curves(vit, perf, string):
    datasets = ["mnist", "fashion_mnist", "cifar10", "cifar100"]
    titles = {
        "mnist": "MNIST",
        "fashion_mnist": "Fashion-MNIST",
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, ds in zip(axes, datasets):
        epochs = range(1, len(vit[ds]["val_acc_curve"]) + 1)

        vit_acc = [x * 100 for x in vit[ds]["val_acc_curve"]]
        perf_acc = [x * 100 for x in perf[ds]["val_acc_curve"]]
        string_acc = [x * 100 for x in string[ds]["val_acc_curve"]]

        ax.plot(epochs, vit_acc, marker="o", label="ViT")
        ax.plot(epochs, perf_acc, marker="s", label="Performer (FAVOR+, r=128)")
        ax.plot(epochs, string_acc, marker="^", label="Performer+STRING (m=128)")

        ax.axhline(vit[ds]["best_val_acc"] * 100, linestyle="--", alpha=0.3)
        ax.axhline(perf[ds]["best_val_acc"] * 100, linestyle="--", alpha=0.3)
        ax.axhline(string[ds]["best_val_acc"] * 100, linestyle="--", alpha=0.3)

        ax.set_title(
            f"{titles[ds]}\n"
            f"Best: ViT={vit[ds]['best_val_acc']*100:.1f}% | "
            f"Perf ={perf[ds]['best_val_acc']*100:.1f}% | "
            f"STRING={string[ds]['best_val_acc']*100:.1f}%"
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Validation Accuracy Learning Curves (L=64, 20 Epochs)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "fig_l64_validation_curves.png", dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# 3. L=256 learning curve
# =========================

def plot_l256_curve(l256):
    models = [
        ("ViT_L256", "ViT L=256", "o"),
        ("Performer_L256", "Performer L=256", "s"),
        ("Performer_STRING_L256", "Performer+STRING L=256", "^"),
    ]

    plt.figure(figsize=(10, 6))

    title_parts = []
    for key, label, marker in models:
        r = l256[key]
        epochs = range(1, len(r["val_acc_curve"]) + 1)
        plt.plot(epochs, [x * 100 for x in r["val_acc_curve"]], marker=marker, label=label)
        plt.axhline(r["best_val_acc"] * 100, linestyle="--", alpha=0.25)
        short_name = label.replace(" L=256", "")
        title_parts.append(f"{short_name}={r['best_val_acc']*100:.1f}%")

    plt.title(
        "Validation Accuracy Learning Curves on CIFAR-10 (L=256, 10 Epochs)\n"
        + "Best: " + " | ".join(title_parts)
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "fig_l256_validation_curve.png", dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# 4. L=256 summary table
# =========================

def make_l256_table(l256):
    display_names = {
        "ViT_L256": "ViT L=256",
        "Performer_L256": "Performer L=256 (r=128)",
        "Performer_STRING_L256": "Performer+STRING L=256",
    }

    rows = []
    for model_key, r in l256.items():
        rows.append({
            "Model": display_names.get(model_key, model_key),
            "Dataset": r["dataset"],
            "Best Val Acc (%)": round(r["best_val_acc"] * 100, 2),
            "Train Time (s)": r["train_time"],
            "Inference Time (s)": r["inf_time"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(BASE_DIR / "table_l256_results.csv", index=False)

    print("\nL=256 Results Table:")
    print(df.to_string(index=False))

    print("\nLaTeX table:")
    print(df.to_latex(index=False, float_format="%.2f"))

    return df


# =========================
# 5. Main L=64 summary table
# =========================

def make_l64_summary_table(vit, perf, string):
    rows = []
    datasets = ["mnist", "fashion_mnist", "cifar10", "cifar100"]

    for ds in datasets:
        rows.append({
            "Dataset": ds,
            "ViT Acc (%)": round(vit[ds]["best_val_acc"] * 100, 2),
            "Performer Acc (%)": round(perf[ds]["best_val_acc"] * 100, 2),
            "STRING Acc (%)": round(string[ds]["best_val_acc"] * 100, 2),
            "ViT Train (s)": vit[ds]["train_time"],
            "Performer Train (s)": perf[ds]["train_time"],
            "STRING Train (s)": string[ds]["train_time"],
            "ViT Inf (s)": vit[ds]["inf_time"],
            "Performer Inf (s)": perf[ds]["inf_time"],
            "STRING Inf (s)": string[ds]["inf_time"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(BASE_DIR / "table_l64_main_results.csv", index=False)

    print("\nL=64 Main Results Table:")
    print(df.to_string(index=False))

    print("\nLaTeX table:")
    print(df.to_latex(index=False, float_format="%.2f"))

    return df


# =========================
# 6. Parse benchmark_results.txt
# =========================

def parse_benchmark_txt(filename="benchmark_results.txt"):
    path = BASE_DIR / filename
    rows = []

    # Try common encodings because your benchmark file may be UTF-8 or UTF-16.
    for encoding in ["utf-8-sig", "utf-16", "utf-8"]:
        try:
            lines = path.read_text(encoding=encoding).splitlines()
            break
        except UnicodeError:
            continue
    else:
        lines = path.read_text(errors="ignore").splitlines()

    for line in lines:
        match = re.match(
            r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+(.+)",
            line,
        )
        if match:
            seq_len, vit_ms, perf_ms, speedup, note = match.groups()
            rows.append({
                "Seq Len": int(seq_len),
                "ViT (ms)": float(vit_ms),
                "Performer (ms)": float(perf_ms),
                "Speedup": float(speedup),
                "Faster Model": note.strip(),
            })

    df = pd.DataFrame(rows)
    df.to_csv(BASE_DIR / "table_attention_benchmark.csv", index=False)

    print("\nAttention Benchmark Table:")
    print(df.to_string(index=False))

    print("\nLaTeX table:")
    print(df.to_latex(index=False, float_format="%.3f"))

    return df


# =========================
# 7. Plot benchmark result
# =========================

def plot_attention_benchmark(df):
    plt.figure(figsize=(9, 6))

    plt.plot(df["Seq Len"], df["ViT (ms)"], marker="o", label="ViT Attention")
    plt.plot(df["Seq Len"], df["Performer (ms)"], marker="s", label="Performer FAVOR+ Attention")

    plt.xscale("log", base=2)
    plt.yscale("log")

    plt.title("Attention-Only Runtime Benchmark")
    plt.xlabel("Sequence Length")
    plt.ylabel("Runtime per Forward Pass (ms)")
    plt.grid(True, alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "fig_attention_benchmark_runtime.png", dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# 8. Plot benchmark speedup
# =========================

def plot_attention_speedup(df):
    plt.figure(figsize=(9, 6))

    plt.plot(df["Seq Len"], df["Speedup"], marker="o", label="ViT Time / Performer Time")
    plt.axhline(1.0, linestyle="--", alpha=0.5)

    plt.xscale("log", base=2)

    plt.title("Performer Attention Speedup over ViT Attention")
    plt.xlabel("Sequence Length")
    plt.ylabel("Speedup Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "fig_attention_benchmark_speedup.png", dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# 9. Run all plots/tables
# =========================

if __name__ == "__main__":
    plot_l64_learning_curves(vit, perf, string)
    l256 = {
    "ViT_L256": vit_256,
    "Performer_L256": performer_256,
    "Performer_STRING_L256": string_256
    }
    plot_l256_curve(l256)
    make_l256_table(l256)
    make_l64_summary_table(vit, perf, string)

    benchmark_path = BASE_DIR / "benchmark_results.txt"
    if benchmark_path.exists():
        benchmark_df = parse_benchmark_txt("benchmark_results.txt")
        plot_attention_benchmark(benchmark_df)
        plot_attention_speedup(benchmark_df)
    else:
        print("\nSkipping attention benchmark plots because benchmark_results.txt was not found.")
