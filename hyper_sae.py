from abc import ABC, abstractmethod
from functools import partial
from dataclasses import dataclass
from einops import einsum, rearrange
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from vis import visualize_weights_2d, visualize_weights_3d

l1 = partial(torch.norm, p=1)
l2 = partial(torch.norm, p=2)


class AutoEncoder(ABC, nn.Module):

    @dataclass
    class Losses:
        sparsity: torch.Tensor
        mse: torch.Tensor

    @abstractmethod
    def forward_train(self, x_NF: torch.Tensor) -> tuple[torch.Tensor, Losses]: ...

    # @abstractmethod def get_effective_weights(self) -> torch.Tensor: ...


class HyperSparseAutoencoder(AutoEncoder):
    """A generalization of an SAE into higher feature dimensions"""

    def __init__(self, n_features: int, n_latents: int, latent_dim: int):
        super().__init__()
        self.n_latents = n_latents
        self.latent_dim = latent_dim

        self.W_FLnLf = nn.Parameter(torch.randn(n_features, n_latents, latent_dim))
        # self.W_dec_LfFLn = nn.Parameter(rearrange(self.W_FLnLf, "n_features latent latent_dim -> latent latent_dim n_features"))

    def encode(self, x_NF: torch.Tensor) -> torch.Tensor:
        latent_NLF = einsum(x_NF, self.W_FLnLf, "n feature, feature latent latent_dim -> n latent latent_dim")
        return latent_NLF

    def decode(self, latent_NLF: torch.Tensor) -> torch.Tensor:
        return einsum(latent_NLF, self.W_FLnLf, "n latent latent_dim, feature latent latent_dim -> n feature")

    def sparsity_loss(self, hidden_NLF: torch.Tensor) -> torch.Tensor:
        """Sparsity is defined as the L1 norm of the magnitudes of the latent features."""
        magnitudes_NL = l2(hidden_NLF, dim=-1)
        return l1(magnitudes_NL, dim=-1).mean()

    def forward_train(self, x_NF: torch.Tensor) -> tuple[torch.Tensor, AutoEncoder.Losses]:
        hidden_NLF = self.encode(x_NF)
        out_ND = self.decode(hidden_NLF)
        losses = self.Losses(
            sparsity=self.sparsity_loss(hidden_NLF),
            mse=F.mse_loss(x_NF, out_ND),
        )
        return out_ND, losses

class SparseAutoencoder(AutoEncoder):
    """A generalization of an SAE into higher feature dimensions"""

    def __init__(self, n_features: int, n_latents: int):
        super().__init__()
        self.n_latents = n_latents

        self.W_FL = nn.Parameter(torch.randn(n_features, n_latents))

    def encode(self, x_NF: torch.Tensor) -> torch.Tensor:
        latent_NL = einsum(x_NF, self.W_FL, "n feature, feature latent -> n latent")
        return latent_NL

    def decode(self, latent_NL: torch.Tensor) -> torch.Tensor:
        return einsum(latent_NL, self.W_FL, "n latent, feature latent -> n feature")

    def sparsity_loss(self, hidden_NL: torch.Tensor) -> torch.Tensor:
        """Sparsity is defined as the L1 norm of the magnitudes of the latent features."""
        return l1(hidden_NL, dim=-1).mean()

    def forward_train(self, x_NF: torch.Tensor) -> tuple[torch.Tensor, AutoEncoder.Losses]:
        hidden_NL = self.encode(x_NF)
        out_ND = self.decode(hidden_NL)
        losses = self.Losses(
            sparsity=self.sparsity_loss(hidden_NL),
            mse=F.mse_loss(x_NF, out_ND),
        )
        return out_ND, losses


@torch.no_grad()
def random_unit(dimension: int):
    unit = torch.randn(dimension)
    return unit / l2(unit)

if __name__ == "__main__":
    n_features = 2
    unit_d = random_unit(n_features)

    visualize_weights_2d(unit_d[None, :].detach().cpu().numpy(), "Unit vector")

    sae = SparseAutoencoder(n_features, n_latents=n_features)
    sae.forward_train(unit_d[None, :])

    h_sae = HyperSparseAutoencoder(n_features, n_latents=1, latent_dim=n_features)
    h_sae.forward_train(unit_d[None, :])

    
