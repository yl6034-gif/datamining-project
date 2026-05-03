import torch
import torch.nn as nn
import math


class FAVORPlusAttention(nn.Module):
    """
    FAVOR+ linear attention from Performer (Choromanski et al., 2020).
    Uses positive orthogonal random features to approximate softmax attention.
    Hyperparameter: num_random_features (r) - set to 256 by default.
    """
    def __init__(self, embed_dim=256, num_heads=8,
                 num_random_features=256, dropout=0.1):
        super().__init__()
        self.num_heads         = num_heads
        self.head_dim          = embed_dim // num_heads
        self.num_random_features = num_random_features

        self.qkv     = nn.Linear(embed_dim, embed_dim * 3)
        self.proj    = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Sample random orthogonal features once at init
        self.register_buffer('random_matrix',
                             self._sample_ortho_features())

    def _sample_ortho_features(self):
        """
        Sample orthogonal random feature matrix W of shape (r, head_dim).
        Uses Gram-Schmidt block orthogonalization for variance reduction.
        """
        r = self.num_random_features
        d = self.head_dim
        blocks = math.ceil(r / d)
        W_list = []
        for _ in range(blocks):
            # Random Gaussian block
            block = torch.randn(d, d)
            # QR decomposition gives orthogonal matrix
            Q, _ = torch.linalg.qr(block)
            # Scale by sqrt(d) for unbiased estimation
            W_list.append(Q * math.sqrt(d))
        W = torch.cat(W_list, dim=0)[:r]   # (r, d)
        return W

    def _phi(self, x):
        """
        Positive random feature map for softmax kernel approximation.
        phi(x) = exp(-||x||^2 / 2) * exp(W x) / sqrt(r)
        All outputs are positive, ensuring stable linear attention.
        """
        # x: (B, H, N, head_dim)
        norm_sq = (x ** 2).sum(dim=-1, keepdim=True) / 2.0  # (B, H, N, 1)
        # Project onto random features: (B, H, N, r)
        proj = torch.einsum('bhnd,rd->bhnr', x, self.random_matrix)
        phi  = torch.exp(proj - norm_sq) / math.sqrt(self.num_random_features)
        return phi   # (B, H, N, r) - all positive

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)                         # each: (B, H, N, head_dim)

        # Map queries and keys to positive random features
        phi_q = self._phi(q)   # (B, H, N, r)
        phi_k = self._phi(k)   # (B, H, N, r)

        # Linear attention via associativity trick - O(N) complexity
        # Instead of computing N x N attention matrix:
        # out = phi_q * (phi_k^T * V) / (phi_q * sum(phi_k))

        # Step 1: compute sum_j phi_k(j)^T v(j) -> (B, H, r, head_dim)
        kv = torch.einsum('bhnr,bhnd->bhrd', phi_k, v)

        # Step 2: compute normalizer sum_j phi_k(j) -> (B, H, r)
        denom = phi_k.sum(dim=2)

        # Step 3: numerator phi_q @ kv -> (B, H, N, head_dim)
        num = torch.einsum('bhnr,bhrd->bhnd', phi_q, kv)

        # Step 4: denominator phi_q @ denom -> (B, H, N, 1)
        den = torch.einsum('bhnr,bhr->bhn', phi_q, denom).unsqueeze(-1)

        out = num / (den + 1e-6)
        out = out.transpose(1, 2).contiguous().reshape(B, N, C)
        return self.dropout(self.proj(out))


class PerformerBlock(nn.Module):
    """Single Performer block: FAVOR+ attention + MLP."""
    def __init__(self, embed_dim=256, num_heads=8,
                 num_random_features=256, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn  = FAVORPlusAttention(embed_dim, num_heads,
                                        num_random_features, dropout)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PerformerViT(nn.Module):
    """
    Performer ViT using FAVOR+ linear attention.
    Drop-in replacement for regular ViT with O(N) attention complexity.
    num_random_features (r=256): controls approximation quality vs speed.
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 num_classes=10, embed_dim=256, depth=6, num_heads=8,
                 num_random_features=256, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        from model_vit import PatchEmbedding
        self.patch_embed = PatchEmbedding(img_size, patch_size,
                                          in_channels, embed_dim)
        num_patches      = self.patch_embed.num_patches

        self.cls_token   = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout     = nn.Dropout(dropout)

        self.blocks      = nn.ModuleList([
            PerformerBlock(embed_dim, num_heads, num_random_features,
                           mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm        = nn.LayerNorm(embed_dim)
        self.head        = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x   = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.dropout(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])


if __name__ == '__main__':
    model = PerformerViT(num_classes=10)
    x     = torch.randn(4, 3, 32, 32)
    out   = model(x)
    print(f"Input:  {list(x.shape)}")
    print(f"Output: {list(out.shape)}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")