import torch
import time
from model_vit import MultiHeadSelfAttention
from model_performer import FAVORPlusAttention

def benchmark_attention(attn_module, x, n_runs=50):
    """Benchmark attention computation time only."""
    device = x.device
    attn_module = attn_module.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = attn_module(x)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = attn_module(x)
    torch.cuda.synchronize()
    end = time.time()

    return (end - start) / n_runs * 1000  # ms per run

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Benchmarking ATTENTION ONLY (forward pass, no training)")
print(f"embed_dim=256, num_heads=8, num_random_features=256\n")

embed_dim = 256
num_heads = 8
num_random_features = 256

print(f"{'Seq Len':>10} {'ViT(ms)':>12} {'Performer(ms)':>15} {'Speedup':>10} {'Note'}")
print('-' * 65)

for seq_len in [64, 128, 256, 512, 1024, 2048]:
    x = torch.randn(8, seq_len, embed_dim).to(device)

    # Model 1: Standard attention (ViT)
    vit_attn = MultiHeadSelfAttention(embed_dim, num_heads)
    vit_time = benchmark_attention(vit_attn, x)

    # Model 2 & 3: FAVOR+ attention (Performer & Performer+STRING)
    # STRING only affects positional encoding, not attention computation
    perf_attn = FAVORPlusAttention(embed_dim, num_heads, num_random_features)
    perf_time = benchmark_attention(perf_attn, x)

    speedup = vit_time / perf_time
    note = "Performer faster" if speedup > 1 else "ViT faster"
    print(f"{seq_len:>10} {vit_time:>12.3f} {perf_time:>15.3f} {speedup:>10.2f}x  {note}")

print(f"\nNote: Model 2 (Performer) and Model 3 (Performer+STRING) share")
print(f"the same FAVOR+ attention. STRING only affects positional encoding.")