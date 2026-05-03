import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                         # (B, embed_dim, H', W')
        x = x.flatten(2)                         # (B, embed_dim, N)
        x = x.transpose(1, 2).contiguous()       # (B, N, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard softmax multi-head self-attention (MPS compatible)."""
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv     = nn.Linear(embed_dim, embed_dim * 3)
        self.proj    = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)                         # each: (B, H, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v)                                 # (B, H, N, head_dim)
        out = out.transpose(1, 2).contiguous().reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Single transformer block: attention + MLP."""
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1  = nn.LayerNorm(embed_dim)
        self.norm2  = nn.LayerNorm(embed_dim)
        self.attn   = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        mlp_dim     = int(embed_dim * mlp_ratio)
        self.mlp    = nn.Sequential(
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


class VisionTransformer(nn.Module):
    """Regular ViT for image classification."""
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 num_classes=10, embed_dim=256, depth=6,
                 num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size,
                                          in_channels, embed_dim)
        num_patches      = self.patch_embed.num_patches

        self.cls_token   = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout     = nn.Dropout(dropout)

        self.blocks      = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
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
        return self.head(x[:, 0])                # CLS token output


if __name__ == '__main__':
    model = VisionTransformer(num_classes=10)
    x     = torch.randn(4, 3, 32, 32)
    out   = model(x)
    print(f"Input:  {list(x.shape)}")
    print(f"Output: {list(out.shape)}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")