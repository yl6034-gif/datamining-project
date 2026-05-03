import torch
import torch.nn as nn
import torch.nn.functional as F
from model_vit import PatchEmbedding

# ---------------------------------------------------------------------------
# Helper: orthogonal random matrix
# ---------------------------------------------------------------------------

def _sample_orth_matrix(m: int, d: int, device: torch.device) -> torch.Tensor:
    """Sample (m, d) orthogonal random matrix via QR decomposition."""
    nb_full_blocks = m // d
    blocks = []
    for _ in range(nb_full_blocks):
        H = torch.randn(d, d)
        Q, _ = torch.linalg.qr(H)
        norms = torch.randn(d).norm()
        blocks.append(Q * norms)
    remainder = m - nb_full_blocks * d
    if remainder > 0:
        H = torch.randn(d, d)
        Q, _ = torch.linalg.qr(H)
        norms = torch.randn(d).norm()
        blocks.append((Q * norms)[:remainder])
    return torch.cat(blocks, dim=0).to(device)


def _favor_kernel(x: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """
    Positive random feature map phi(x).
    x: (..., d), proj: (m, d)
    Returns: (..., m)
    """
    d = x.shape[-1]
    x_scaled = x * (d ** -0.25)
    proj_x = x_scaled @ proj.T
    norm_sq = (x_scaled ** 2).sum(dim=-1, keepdim=True) / 2.0
    return torch.exp(proj_x - norm_sq)


def _linear_attention(q_feat, k_feat, v, eps=1e-6):
    """
    Bidirectional linear attention via FAVOR+.
    q_feat, k_feat: (B, H, N, m)
    v: (B, H, N, D)
    """
    context = torch.einsum('bhnm,bhne->bhme', k_feat, v)   # (B, H, m, D)
    num = torch.einsum('bhnm,bhme->bhne', q_feat, context)  # (B, H, N, D)
    k_sum = k_feat.sum(dim=2)                               # (B, H, m)
    den = torch.einsum('bhnm,bhm->bhn', q_feat, k_sum).unsqueeze(-1).clamp(min=eps)
    return num / den


# ---------------------------------------------------------------------------
# Circulant-STRING
# ---------------------------------------------------------------------------

class CirculantSTRING(nn.Module):
    """
    Circulant-STRING positional encoding (Schenck et al., 2025).
    Applied to Q and K before FAVOR+ attention (correct per paper).
    Uses FFT for O(d log d) computation per Theorem 3.5.
    """
    def __init__(self, num_heads: int, head_dim: int,
                 height: int, width: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.height = height
        self.width = width

        # Learnable circulant vectors per head per axis
        self.c_row = nn.Parameter(torch.zeros(num_heads, head_dim))
        self.c_col = nn.Parameter(torch.zeros(num_heads, head_dim))
        nn.init.normal_(self.c_row, std=0.02)
        nn.init.normal_(self.c_col, std=0.02)

        # Precompute 2D patch positions
        rows = torch.arange(height, dtype=torch.float32)
        cols = torch.arange(width, dtype=torch.float32)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')
        positions = torch.stack([
            grid_r.flatten(), grid_c.flatten()
        ], dim=-1)  # (N, 2)
        self.register_buffer('positions', positions, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_heads, N, head_dim)
        Returns:
            (B, num_heads, N, head_dim) with STRING rotation applied
        """
        B, H, N, D = x.shape
        pos = self.positions  # (N, 2)

        # g(r_i) = row_i * c_row + col_i * c_col → (H, N, D)
        g = (pos[:, 0:1].unsqueeze(0) * self.c_row.unsqueeze(1) +
             pos[:, 1:2].unsqueeze(0) * self.c_col.unsqueeze(1))

        # FFT eigenvalues of antisymmetric circulant: 2i * Im(FFT(g))
        g_fft = torch.fft.fft(g.float().contiguous(), dim=-1)
        lambda_k = 2j * g_fft.imag
        exp_lambda = torch.exp(lambda_k)  # unit complex

        # Apply rotation: IFFT(exp_lambda * FFT(x))
        x_fft = torch.fft.fft(x.float().contiguous(), dim=-1)
        rotated_fft = exp_lambda.unsqueeze(0) * x_fft
        rotated = torch.fft.ifft(rotated_fft, dim=-1).real

        return rotated.to(x.dtype)


# ---------------------------------------------------------------------------
# FAVOR+ Attention with Circulant-STRING
# ---------------------------------------------------------------------------

REDRAW_STEPS = 1000

class FAVORCirculantSTRINGAttention(nn.Module):
    """
    FAVOR+ attention with Circulant-STRING relative positional encoding.
    STRING is applied to Q and K before FAVOR+ kernel (correct per paper).
    CLS token is left unrotated (no spatial position).
    """
    NUM_RANDOM_FEATURES_MULTIPLIER = 4  # m = 4 * head_dim

    def __init__(self, embed_dim: int, num_heads: int,
                 height: int, width: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.m = self.NUM_RANDOM_FEATURES_MULTIPLIER * self.head_dim

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.string = CirculantSTRING(num_heads, self.head_dim, height, width)

        self.register_buffer('_proj', None, persistent=False)
        self._step_count = 0

    def _get_projection(self, device):
        if self._proj is None or self._step_count % REDRAW_STEPS == 0:
            proj = _sample_orth_matrix(self.m, self.head_dim, device)
            if self._proj is None:
                self._proj = proj
            else:
                self._proj.copy_(proj)
        return self._proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._step_count += 1

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, H, N, D)

        # Separate CLS token from patch tokens
        has_cls = (N == self.string.height * self.string.width + 1)
        if has_cls:
            q_cls, q_patches = q[:, :, :1, :], q[:, :, 1:, :]
            k_cls, k_patches = k[:, :, :1, :], k[:, :, 1:, :]
        else:
            q_patches, k_patches = q, k

        # Apply STRING to patch tokens only (on Q and K)
        q_patches = self.string(q_patches)
        k_patches = self.string(k_patches)

        if has_cls:
            q = torch.cat([q_cls, q_patches], dim=2)
            k = torch.cat([k_cls, k_patches], dim=2)

        # FAVOR+ linear attention
        proj = self._get_projection(x.device)
        q_feat = _favor_kernel(q, proj)
        k_feat = _favor_kernel(k, proj)

        out = _linear_attention(q_feat, k_feat, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Performer Block with STRING
# ---------------------------------------------------------------------------

class PerformerSTRINGBlock(nn.Module):
    """Transformer block with FAVOR+ + Circulant-STRING attention."""
    def __init__(self, embed_dim, num_heads, height, width,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = FAVORCirculantSTRINGAttention(
            embed_dim, num_heads, height, width, dropout
        )
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
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


# ---------------------------------------------------------------------------
# Full Performer ViT with Circulant-STRING
# ---------------------------------------------------------------------------

class PerformerViTSTRING(nn.Module):
    """
    Performer ViT with Circulant-STRING positional encoding.
    STRING is applied correctly to Q and K (not token embeddings).
    No absolute positional embedding (STRING provides relative pos enc).
    Based on Schenck et al. (2025) Definition 3.1 and Theorem 3.5.
    """
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 num_classes=10, embed_dim=256, depth=6, num_heads=8,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size,
                                          in_channels, embed_dim)
        grid_size = img_size // patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # No absolute pos embed for STRING variant
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            PerformerSTRINGBlock(embed_dim, num_heads, grid_size, grid_size,
                                 mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight init
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])


if __name__ == '__main__':
    print("Testing Performer + Circulant-STRING (correct implementation)...")
    model = PerformerViTSTRING(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"Input:  {list(x.shape)}")
    print(f"Output: {list(out.shape)}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")