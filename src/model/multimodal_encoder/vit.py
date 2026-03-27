import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dropout_p = dropout

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, img_size, patch_size, dim, depth, heads, mlp_dim, channels = 1, dim_head = 64, dropout = 0., emb_dropout = 0., use_cls = False):
        super().__init__()
        image_height, image_width, img_depth = img_size
        patch_height, patch_width, patch_depth = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert img_depth % patch_depth == 0, 'Depth must be divisible by patch depth'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (img_depth // patch_depth)
        patch_dim = channels * patch_height * patch_width * patch_depth

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h ph) (w pw) (d pd) -> b (h w d) (ph pw pd c)', ph = patch_height, pw = patch_width, pd = patch_depth),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.use_cls = use_cls
        if use_cls:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.norm = nn.LayerNorm(dim)
        self.hidden_size = dim

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        if self.use_cls:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
        else:
            x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x)

        return x
