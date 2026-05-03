import torch
import torch.nn as nn
from model_performer import PerformerBlock
from model_vit import PatchEmbedding


class CirculantSTRING(nn.Module):
    """
    Circulant-STRING position encoding (Schenck et al., 2025).
    Based on Theorem 3.5: computed via FFT in O(d log d) time.
    For 2D image patches, dc=2 (row, col coordinates).
    """
    def __init__(self, dim, dc=2):
        super().__init__()
        self.dim = dim
        self.dc = dc
        # Learnable circulant parameters per position dimension
        self.circulant_params = nn.ParameterList([
            nn.Parameter(torch.randn(dim) * 0.01)
            for _ in range(dc)
        ])

    def _apply_single(self, z, c):
        """
        Apply circulant STRING to token z: (B, dim).
        Per Theorem 3.5: p = DFT * diag(exp(DFT * u)) * iDFT * z
        """
        D = z.shape[-1]
        c_fft = torch.fft.rfft(c, n=D)
        u = 2 * c_fft.imag
        z_fft = torch.fft.rfft(z, n=D)
        exp_u = torch.complex(torch.cos(u), torch.sin(u))
        out_fft = exp_u * z_fft
        return torch.fft.irfft(out_fft, n=D)

    def forward(self, x, positions):
        """
        x: (B, N, dim)
        positions: (N, 2) 2D grid positions
        """
        B, N, D = x.shape
        # Use list to avoid inplace operations
        out_list = [x[:, i, :] for i in range(N)]

        for k in range(self.dc):
            pos_k = positions[:, k].float()
            c = self.circulant_params[k]
            scaled_c = c.unsqueeze(0) * pos_k.unsqueeze(1)  # (N, dim)
            for i in range(N):
                out_list[i] = self._apply_single(out_list[i], scaled_c[i])

        out = torch.stack(out_list, dim=1)  # (B, N, dim)
        return out


class PerformerViTSTRING(nn.Module):
    """
    Performer ViT with Circulant-STRING positional encoding.
    Combines FAVOR+ linear attention with STRING relative position encoding.
    Based on Schenck et al. (2025) Definition 3.1 and Theorem 3.5.
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 num_classes=10, embed_dim=256, depth=6, num_heads=8,
                 num_random_features=256, mlp_ratio=4.0, dropout=0.1,
                 string_type='circulant'):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size,
                                          in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        grid_size = img_size // patch_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cls_pos_embed = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Circulant-STRING positional encoding
        self.string = CirculantSTRING(embed_dim, dc=2)

        # Precompute 2D patch grid positions
        rows = torch.arange(grid_size).float()
        cols = torch.arange(grid_size).float()
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')
        positions = torch.stack([
            grid_r.flatten(), grid_c.flatten()
        ], dim=1)  # (num_patches, 2)
        self.register_buffer('positions', positions)

        self.blocks = nn.ModuleList([
            PerformerBlock(embed_dim, num_heads, num_random_features,
                           mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)             # (B, N, dim)
        x = self.string(x, self.positions)  # Apply STRING
        cls = self.cls_token.expand(B, -1, -1) + self.cls_pos_embed
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])


if __name__ == '__main__':
    print("Testing Performer + Circulant-STRING...")
    model = PerformerViTSTRING(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"Input:  {list(x.shape)}")
    print(f"Output: {list(out.shape)}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")