import torch
import torch.nn as nn
import numpy as np


def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(
        n, n_patches**2, h * w * c // n_patches**2, device=images.device
    )
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


def get_positional_embeddings(sequence_length, token_dim):
    result = torch.ones(sequence_length, token_dim)
    for i in range(sequence_length):
        for j in range(token_dim):
            result[i][j] = (
                np.sin(i / (10000 ** (j / token_dim)))
                if j % 2 == 0
                else np.cos(i / (10000 ** ((j - 1) / token_dim)))
            )
    return result


class MSA(nn.Module):
    """Multi-head Self Attention"""

    def __init__(self, token_dim, n_heads=2, use_linear_mapping: bool = True):
        super().__init__()
        self._token_dim = token_dim
        self._n_heads = n_heads
        self._use_linear_mapping = use_linear_mapping

        assert (
            token_dim % n_heads == 0
        ), f"Can't divide dimension {token_dim} into {n_heads} heads"

        d_head = int(token_dim / n_heads)
        d_head_in = token_dim if self._use_linear_mapping else d_head
        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head_in, d_head) for _ in range(self._n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head_in, d_head) for _ in range(self._n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head_in, d_head) for _ in range(self._n_heads)]
        )
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        # NOTE: Not vectorized for readability (could vectorize for improved performance)
        for sequence in sequences:
            seq_result = []
            for head in range(self._n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                if not self._use_linear_mapping:
                    sequence = sequence[
                        :, head * self.d_head : (head + 1) * self.d_head
                    ]
                q, k, v = q_mapping(sequence), k_mapping(sequence), v_mapping(sequence)

                attention = self.softmax(q @ k.T / (self.d_head**0.5))
                seq_result.append(
                    attention @ v
                )  # Shape (N, seq_length, n_heads, token_dim / n_heads)
            result.append(torch.hstack(seq_result))
        return torch.cat(
            [torch.unsqueeze(r, dim=0) for r in result]
        )  # Shape (N, seq_length, token_dim)


class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super().__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.msa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.msa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class ViT(nn.Module):
    def __init__(
        self,
        chw=(1, 28, 28),
        n_patches=7,
        n_blocks=2,
        hidden_d=8,
        n_heads=2,
        out_d=10,
        use_mean_pooling=False,
    ):
        super().__init__()

        # Attributes
        self._chw = chw  # (C, H, W)
        self._n_patches = n_patches
        self._n_blocks = n_blocks
        self._n_heads = n_heads
        self._hidden_d = hidden_d
        self._use_mean_pooling = use_mean_pooling

        assert (
            chw[1] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        assert (
            chw[2] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self._hidden_d)

        # 2) Learnable classifiation token
        self.class_token = nn.Parameter(torch.rand(1, self._hidden_d))

        # 3) Positional embedding
        self.register_buffer(
            "positional_embeddings",
            get_positional_embeddings(n_patches**2 + 1, hidden_d),
            persistent=False,
        )

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )

        # 5) Classification MLP
        self.classification_mlp = nn.Sequential(
            nn.Linear(self._hidden_d, out_d), nn.Softmax(dim=-1)
        )

    def forward(self, images):
        patches = patchify(images, self._n_patches)
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.stack(
            [torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))]
        )

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(len(tokens), 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        out = out.mean(dim=1) if self._use_mean_pooling else out[:, 0]

        # Map to output dimension, output category distribution
        return self.classification_mlp(out)
