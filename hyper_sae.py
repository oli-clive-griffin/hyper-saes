from abc import ABC, abstractmethod
from functools import partial
from dataclasses import dataclass
from einops import einsum, rearrange
import numpy as np
import torch
from torch import nn

l1 = partial(torch.norm, p=1)
l2 = partial(torch.norm, p=2)


class AutoEncoder(ABC, nn.Module):
    W_dec_HD: torch.Tensor
    b_dec_D: torch.Tensor

    @dataclass
    class Losses:
        sparsity: torch.Tensor
        reconstruction: torch.Tensor

    @abstractmethod
    def forward_train(self, x_BD: torch.Tensor) -> tuple[torch.Tensor, Losses]: ...

    @torch.no_grad()
    def enforce_decoder_norm(self):
        self.W_dec_HD.data = self.W_dec_HD / l2(self.W_dec_HD, dim=-1, keepdim=True)

    @abstractmethod
    def sparsity_loss(self, hidden_BHF: torch.Tensor) -> torch.Tensor: ...

    def reconstruction_loss(self, x_BD, out_BD):
        return torch.mean(torch.square(x_BD - out_BD))

    @abstractmethod
    def get_effective_weights(self) -> np.ndarray: ...


class HyperSparseAutoencoder(AutoEncoder):
    """A generalization of an SAE into higher feature dimensions"""

    def __init__(self, d_model: int, hidden_features: int, feature_dim: int):
        super().__init__()
        self.hidden_features = hidden_features
        self.feature_dim = feature_dim

        self.W_enc_DHF = nn.Parameter(torch.randn(d_model, hidden_features, feature_dim))
        with torch.no_grad():
            self.W_enc_DHF.data = self.W_enc_DHF / l2(self.W_enc_DHF, dim=-1, keepdim=True)

        self.W_dec_DHF = nn.Parameter(rearrange(self.W_enc_DHF, "d_model hidden feat_dim -> d_model feat_dim hidden"))

    def encode(self, x_BD: torch.Tensor) -> torch.Tensor:
        z_BHF = einsum(x_BD, self.W_enc_DHF, "b d_model, d_model hidden feat_dim -> b hidden feat_dim")

        # Unit vectors in feature direction
        feature_dir_subspace_BH1 = l2(z_BHF, dim=-1, keepdim=True)
        direction_unit_BHF = z_BHF / (feature_dir_subspace_BH1 + 1e-8)

        # Add bias along feature direction
        # bias_BHF = einsum(
        #     direction_unit_BHF,
        #     self.b_enc_H,
        #     "b hidden feat_dim, hidden -> b hidden feat_dim",
        # )
        # z_BHF = z_BHF + bias_BHF

        # Project onto feature direction to get magnitude, can be negative.
        # for each hidden feature, this is just a dot product between the full-rank feature
        # and the unit vector in the feature direction
        mag_BH = einsum(
            # z_BHF,
            direction_unit_BHF,
            "b hidden feat_dim, b hidden feat_dim -> b hidden",
        )

        # ReLU in magnitude space
        mag_BH1 = torch.relu(mag_BH)[:, :, None]

        # apply this in vector space by scaling the direction unit vector
        out_BHF = direction_unit_BHF * mag_BH1

        return out_BHF

    def decode(self, hidden_BHF: torch.Tensor) -> torch.Tensor:
        return torch.relu(hidden_BHF @ self.W_dec_HD)  # + self.b_dec_D)

    def forward_train(self, x_BD: torch.Tensor) -> tuple[torch.Tensor, AutoEncoder.Losses]:
        hidden_BHF = self.encode(x_BD)
        out_BD = self.decode(rearrange(hidden_BHF, "b hidden feat_dim -> b (hidden feat_dim)"))
        losses = self.Losses(
            sparsity=self.sparsity_loss(hidden_BHF),
            reconstruction=self.reconstruction_loss(x_BD, out_BD),
        )
        return out_BD, losses

    def sparsity_loss(self, hidden_BHF: torch.Tensor) -> torch.Tensor:
        """Sparsity is defined as the L1 norm of the magnitudes of the hidden features."""
        magnitudes_BH = l2(hidden_BHF, dim=-1)
        return l1(magnitudes_BH, dim=-1).mean()

    def get_effective_weights(self) -> np.ndarray:
        W_in_DHf = rearrange(self.W_enc_DH, "d_model hidden feat_dim -> d_model (hidden feat_dim)")
        return W_in_DHf.detach().cpu().numpy()


class SparseAutoencoder(AutoEncoder):
    """A generalization of an SAE into higher feature dimensions"""

    def __init__(self, d_model: int, hidden_features: int):
        super().__init__()
        self.hidden_features = hidden_features

        self.W_enc_DH = nn.Parameter(torch.randn(d_model, hidden_features))
        with torch.no_grad():
            self.W_enc_DH.data = self.W_enc_DH / l2(self.W_enc_DH, dim=-1, keepdim=True)

        # self.b_enc_H = nn.Parameter(torch.zeros(hidden_features))

        self.W_dec_HD = nn.Parameter(torch.randn(hidden_features, d_model))
        # with torch.no_grad():
        #     self.W_dec_HD.data = self.W_dec_HD / l2(self.W_dec_HD, dim=-1, keepdim=True)
        # self.b_dec_D = nn.Parameter(torch.zeros(d_model))

    def encode(self, x_BD: torch.Tensor) -> torch.Tensor:
        return torch.relu(
            einsum(
                x_BD,
                self.W_enc_DH,
                "b d_model, d_model hidden -> b hidden",
            )
            # + self.b_enc_H
        )

    def sparsity_loss(self, hidden_BH: torch.Tensor) -> torch.Tensor:
        """Sparsity is defined as the L1 norm of the magnitudes of the hidden features."""
        return l1(hidden_BH, dim=-1).mean()

    def get_effective_weights(self) -> np.ndarray:
        return self.W_enc_DH.detach().cpu().numpy()


def sanity_check(model: AutoEncoder, d_model: int):
    batch_size = 10
    x_BD = torch.randn(batch_size, d_model)

    out_BD, losses = model.forward_train(x_BD)
    print(out_BD.shape, losses)


def unit():
    unit = torch.randn(3)
    with torch.no_grad():
        unit = unit / l2(unit, dim=-1, keepdim=True)
    return unit


if __name__ == "__main__":
    unit_d = unit()

    W_DH = torch.randn(3, 4)
    with torch.no_grad():
        W_DH_1 = W_DH / l2(W_DH, dim=1, keepdim=True)

    with torch.no_grad():
        W_DH_2 = W_DH / l2(W_DH, dim=0, keepdim=True)

    out = einsum(unit_d, W_DH_1, "d_model, d_model hidden -> hidden")
    out_ = einsum(unit_d, W_DH_2, "d_model, d_model hidden -> hidden")

    print(out.norm())
    print(out_.norm())

    # d_model = 4

    # h_sae = HyperSparseAutoencoder(d_model, hidden_features=2, feature_dim=2)
    # sanity_check(h_sae, d_model)

    # sae = SparseAutoencoder(d_model, hidden_features=4)
    # sanity_check(sae, d_model)
