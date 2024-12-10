from functools import partial
from dataclasses import dataclass
from einops import einsum
import torch
from torch import nn

l1 = partial(torch.norm, p=1)
l2 = partial(torch.norm, p=2)


class NSAE(nn.Module):
    """A generalizaion of an SAE into higher feature dimensions.

    where
    """

    @dataclass
    class Losses:
        sparsity: torch.Tensor
        reconstruction: torch.Tensor

    def __init__(self, d_model: int, hidden_features: int, feature_dim: int, bias: bool = True):
        super().__init__()
        self.W_enc_DHF = nn.Parameter(torch.randn(d_model, hidden_features, feature_dim))
        self.b_enc_H = nn.Parameter(torch.randn(hidden_features))

        self.W_dec_HFD = nn.Parameter(torch.randn(hidden_features, feature_dim, d_model))
        self.b_dec_D = nn.Parameter(torch.randn(d_model))

    def encode(self, x_BD: torch.Tensor) -> torch.Tensor:
        # Your implementation with added comments and slight optimization
        z_BHF = einsum(
            x_BD,
            self.W_enc_DHF,
            "b d_model, d_model hidden feat_dim -> b hidden feat_dim",
        )

        # Unit vectors in feature direction
        magnitudes_BH1 = l2(z_BHF, dim=-1, keepdim=True)
        direction_unit_BHF = z_BHF / (magnitudes_BH1 + 1e-8)

        # Add bias along feature direction
        bias_BHF = einsum(
            direction_unit_BHF,
            self.b_enc_H,
            "b hidden feat_dim, hidden -> b hidden feat_dim",
        )
        z_BHF = z_BHF + bias_BHF

        # Project onto feature direction to get magnitude, can be negative.
        # for each hidden feature, this is just a dot product between the full-rank feature
        # and the unit vector in the feature direction
        mag_BH = einsum(
            z_BHF,
            direction_unit_BHF,
            "b hidden feat_dim, b hidden feat_dim -> b hidden",
        )

        # ReLU in magnitude space
        mag_BH1 = torch.relu(mag_BH)[:, :, None]

        # apply this in vector space by scaling the direction unit vector
        out_BHF = direction_unit_BHF * mag_BH1

        return out_BHF

    def decode(self, hidden_BHF: torch.Tensor) -> torch.Tensor:
        z_BD = einsum(
            hidden_BHF,
            self.W_dec_HFD,
            "b hidden feat_dim, hidden feat_dim d_model -> b d_model",
        )

        # alternative impl for inline asserting
        self.__assert_impl(hidden_BHF, z_BD)

        return torch.relu(z_BD + self.b_dec_D)

    def __assert_impl(self, hidden_BHF, z_BD):
        _z_BD = torch.zeros_like(z_BD)
        for f_idx in range(self.W_enc_DHF.shape[-1]):
            hidden_BH = hidden_BHF[:, :, f_idx]
            W_dec_HD = self.W_dec_HFD[:, f_idx, :]
            _z_BD += hidden_BH @ W_dec_HD

        assert torch.allclose(_z_BD, z_BD)

    def reconstruction_loss(self, x_BD, out_BD):
        return torch.norm(x_BD - out_BD, dim=-1).mean()

    def sparsity_loss(self, hidden_BHF: torch.Tensor) -> torch.Tensor:
        """Sparsity is defined as the L1 norm of the magnitudes of the hidden features."""
        magnitudes_BH = l2(hidden_BHF, dim=-1)
        return l1(magnitudes_BH, dim=-1).mean()

    def forward_train(self, x_BD: torch.Tensor) -> tuple[torch.Tensor, Losses]:
        hidden_BHF = self.encode(x_BD)
        out_BD = self.decode(hidden_BHF)

        losses = self.Losses(
            sparsity=self.sparsity_loss(hidden_BHF),
            reconstruction=self.reconstruction_loss(x_BD, out_BD),
        )

        return out_BD, losses


if __name__ == "__main__":
    d_model = 10
    hidden_features = 10
    feature_dim = 10
    x_BD = torch.randn(10, 10)
    nsae = NSAE(d_model, hidden_features, feature_dim)
    out_BD, losses = nsae.forward_train(x_BD)
    print(out_BD.shape)
