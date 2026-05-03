# datamining-project

```markdown
# Efficient Vision Transformer Models

## Overview
Comparison of Regular ViT, Performer (FAVOR+), and Performer+Circulant-STRING 
on image classification tasks (CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST).

## Models
- **Model 1 (ViT)**: Regular Vision Transformer with standard softmax attention O(N²)
- **Model 2 (Performer)**: FAVOR+ linear attention O(N), Choromanski et al. (2020)
- **Model 3 (Performer+STRING)**: FAVOR+ attention + Circulant-STRING positional encoding, Schenck et al. (2025)

## Requirements
```
pip install torch torchvision timm einops datasets
```

## File Structure
```
├── data_loader.py                   # Dataset loading (HuggingFace)
├── model_vit.py                     # Model 1: Regular ViT
├── model_performer.py               # Model 2: Performer (FAVOR+)
├── model_performer_string.py        # Model 3: Performer + Circulant-STRING
├── train.py                         # Training and evaluation pipeline
├── run_vit.py                       # Run Model 1 on all 4 datasets
├── run_performer.py                 # Run Model 2 on all 4 datasets
├── run_performer_string.py          # Run Model 3 on all 4 datasets
├── run_L256_10epoch_comparison.py   # L=256 comparison (ViT vs Performer)
├── benchmark_attention.py           # Attention-only speed benchmark
└── results/                         # Experiment results (JSON)
```

## Usage
```bash
# Main experiments (L=64, 20 epochs)
python run_vit.py
python run_performer.py
python run_performer_string.py

# L=256 comparison (10 epochs, CIFAR-10 only)
python run_L256_10epoch_comparison.py

# Attention benchmark
python benchmark_attention.py
```

## Experiment Summary

### Block 1: Main Experiments (L=64, All 3 Models, 4 Datasets)
**Setup:**
- img_size=32, patch_size=4 → sequence length L=(32/4)²=64
- epochs=20, batch_size=128, lr=3e-4, optimizer=AdamW
- Hardware: RTX 4060 Laptop GPU, PyTorch 2.5.1, CUDA 12.5

**Models:**
- Model 1 (ViT): Standard softmax attention O(N²), absolute positional encoding
- Model 2 (Performer): FAVOR+ linear attention O(N), r=256 random features, orthogonal features, with redrawing per epoch
- Model 3 (Performer+STRING): FAVOR+ attention (m=128) with Circulant-STRING 
  applied to Q and K before attention (Schenck et al., 2025). No absolute 
  positional embedding. STRING parameters initialized with N(0, 0.02).

**Result files:**
```
results_vit_all.json
results_performer_all.json
results_string_all.json
```

**Key findings:**
- At L=64, Performer is slower than ViT due to FAVOR+ overhead
- Performer accuracy slightly lower than ViT on all datasets
- CIFAR-10 shows largest accuracy gap (-5.44%)

---

### Block 2: Sequence Length Comparison (L=256, CIFAR-10 only)
**Setup:**
- img_size=64, patch_size=4 → sequence length L=(64/4)²=256
- epochs=10 (reduced due to computational constraints)
- Only CIFAR-10 dataset, only ViT and Performer

**Result files:**
```
results_L256_comparison.json
```

**Key findings:**
- ViT L=256: val_acc=70.59%, train_time=1942.5s
- Performer L=256: val_acc=60.27%, train_time=17683.8s
- Performer is 9x slower than ViT at L=256
- Contrary to theoretical expectation (should be 4x faster)
- Reason: PyTorch FAVOR+ lacks hardware-specific CUDA optimization unlike JAX/TPU implementation in original paper

---

### Block 3: Attention-Only Benchmark
**Setup:**
- Isolated attention module benchmark (no training, no MLP, no data loading)
- batch_size=8, embed_dim=256, num_heads=8, num_random_features=256
- 50 runs per configuration, CUDA synchronized timing

**Result file:**
```
benchmark_results.txt
```

**Key findings:**

| Seq Len | ViT(ms) | Performer(ms) | Speedup |
|---|---|---|---|
| 64 | 1.031 | 2.678 | 0.39x |
| 128 | 1.015 | 2.029 | 0.50x |
| 256 | 3.821 | 6.829 | 0.56x |
| 512 | 7.577 | 10.021 | 0.76x |
| 1024 | 25.064 | 13.137 | 1.91x |
| 2048 | 81.177 | 25.472 | 3.19x |

- FAVOR+ attention becomes faster than standard attention only at L≥1024
- At L=64, FAVOR+ is 2.6x slower due to random feature computation overhead
- Note: Model 2 and Model 3 share the same FAVOR+ attention. STRING only affects positional encoding

---

## Important Implementation Notes

**Fair comparison:**
- All models: depth=6, embed_dim=256, num_heads=8, mlp_ratio=4.0
- Same training hyperparameters and hardware for all experiments

把这部分替换成：

```markdown
**Implementation notes:**
- No official PyTorch implementation exists for image classification with Performers
- Official code targets autoregressive language modeling (unidirectional attention)
- We implement bidirectional FAVOR+ following Equation 4 of Choromanski et al. (2020)
- No official STRING implementation is publicly available
- We implement Circulant-STRING following Theorem 3.5 and Definition 3.1 of Schenck et al. (2025)
- Original Performer paper used JAX on TPUs; our PyTorch implementation on CUDA explains the observed performance difference
- We implement **Circulant-STRING** (not Cayley-STRING) because:
  (1) Circulant-STRING achieves the best ImageNet accuracy (81.22%) per Table 1 of Schenck et al. (2025)
  (2) Efficient O(d log d) FFT implementation per Theorem 3.5
  (3) Cayley-STRING requires matrix inverse computation (I+S)^{-1} which is more expensive
- STRING is applied to **queries (Q) and keys (K)** before FAVOR+ attention, following Schenck et al. (2025). CLS token is left unrotated (no spatial position)
- Performer+STRING does **not** use absolute positional embeddings; Circulant-STRING provides relative positional encoding
- Random features redrawn every 1,000 training steps to reduce estimator bias
- num_random_features m = 4 × head_dim = 128 per head
- CirculantSTRING uses vectorized FFT operations for efficiency (no Python for-loops)
```

## References
- Choromanski et al. (2020): Rethinking Attention with Performers. ICLR 2021. arXiv:2009.14794
- Schenck et al. (2025): Learning the RoPEs: Better 2D and 3D Position Encodings with STRING. arXiv:2502.02562
```
