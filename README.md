# datamining-project

这是一个很好的总结，我来写：

---

# Experiment Summary

## Block 1: Main Experiments (L=64, All 3 Models, 4 Datasets)

**Setup:**
- img_size=32, patch_size=4 → sequence length L=(32/4)²=64
- epochs=20, batch_size=128, lr=3e-4, optimizer=AdamW
- Hardware: RTX 4060 Laptop GPU, PyTorch 2.5.1, CUDA 12.5

**Models:**
- Model 1 (ViT): Standard softmax attention O(N²), absolute positional encoding
- Model 2 (Performer): FAVOR+ linear attention O(N), r=256 random features, orthogonal features, with redrawing per epoch
- Model 3 (Performer+STRING): FAVOR+ attention + Circulant-STRING relative positional encoding, based on Theorem 3.5 of Schenck et al. (2025)

**Result files:**
```
results_vit_mnist.json
results_vit_fashion_mnist.json
results_vit_cifar10.json
results_vit_cifar100.json
results_vit_all.json          ← merged

results_performer_mnist.json
results_performer_fashion_mnist.json
results_performer_cifar10.json
results_performer_cifar100.json
results_performer_all.json    ← merged

results_string_mnist.json
results_string_fashion_mnist.json
results_string_cifar10.json
results_string_cifar100.json
results_string_all.json       ← merged
```

**Key findings:**
- At L=64, Performer is slower than ViT due to FAVOR+ overhead
- Performer accuracy slightly lower than ViT on all datasets
- CIFAR-10 shows largest accuracy gap (-5.44%)

---

## Block 2: Sequence Length Comparison (L=256, CIFAR-10 only)

**Setup:**
- img_size=64, patch_size=4 → sequence length L=(64/4)²=256
- epochs=10 (reduced due to computational constraints)
- Same hardware and hyperparameters as Block 1
- Only CIFAR-10 dataset, only ViT and Performer (no STRING)

**Result files:**
```
results_vit_cifar10_L256.json
results_performer_cifar10_L256.json
results_L256_comparison.json  ← combined
```

**Key findings:**
- ViT L=256: val_acc=70.59%, train_time=1942.5s
- Performer L=256: val_acc=60.27%, train_time=17683.8s
- Performer is 9x slower than ViT at L=256
- Contrary to theoretical expectation (should be 4x faster)
- Reason: PyTorch FAVOR+ lacks hardware-specific CUDA optimization unlike JAX/TPU implementation in original paper

---

## Block 3: Attention-Only Benchmark

**Setup:**
- Isolated attention module benchmark (no training, no MLP, no data loading)
- batch_size=8, embed_dim=256, num_heads=8, num_random_features=256
- 50 runs per configuration, CUDA synchronized timing
- Sequence lengths: 64, 128, 256, 512, 1024, 2048

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
- At L=64 (our main experiment), FAVOR+ is 2.6x slower
- This explains why total training time is higher for Performer at L=64
- Note: Model 2 (Performer) and Model 3 (Performer+STRING) share the same FAVOR+ attention. STRING only affects positional encoding, not attention computation

---

## Important Notes

**Fair comparison conditions:**
- All models use identical architecture: depth=6, embed_dim=256, num_heads=8, mlp_ratio=4.0
- Same training hyperparameters across all models
- Same hardware (RTX 4060) for all experiments

**Implementation notes:**
- No official PyTorch implementation exists for image classification with Performers
- Official code targets autoregressive language modeling (unidirectional attention)
- We implement bidirectional FAVOR+ following Equation 4 of Choromanski et al. (2020)
- No official STRING implementation is publicly available
- We implement Circulant-STRING following Theorem 3.5 and Definition 3.1 of Schenck et al. (2025)
- Original Performer paper used JAX on TPUs; our PyTorch implementation on CUDA explains the observed performance difference

---

模型3跑完发我结果，我们就可以开始画图了 👇
