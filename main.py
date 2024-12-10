from functools import partial
from dataclasses import dataclass
from einops import einsum, rearrange
import torch
from torch import nn

l1 = partial(torch.norm, p=1)
l2 = partial(torch.norm, p=2)


class HyperSparseAutoencoder(nn.Module):
    """A generalization of an SAE into higher feature dimensions"""

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
        return torch.relu(z_BD + self.b_dec_D)

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


def test_decode():
    batch_size = 2
    d_model = 10
    hidden_features = 16
    feature_dim = 3

    sae = HyperSparseAutoencoder(d_model, hidden_features, feature_dim)

    def inneficient_correct_decode(hidden_BHF: torch.Tensor) -> torch.Tensor:
        hidden_BHf = rearrange(hidden_BHF, "b hidden feat_dim -> b (hidden feat_dim)")
        w_dec_HD = rearrange(sae.W_dec_HFD, "hidden feat_dim d_model -> (hidden feat_dim) d_model")
        _out_BD = hidden_BHf @ w_dec_HD
        return torch.relu(_out_BD + sae.b_dec_D)

    hidden_BHF = torch.randn(batch_size, hidden_features, feature_dim)

    out_BD = sae.decode(hidden_BHF)
    out_BD_ = inneficient_correct_decode(hidden_BHF)

    assert torch.allclose(out_BD, out_BD_)

def sanity_check():
    d_model = 10
    hidden_features = 10
    feature_dim = 10
    x_BD = torch.randn(10, 10)
    sae = HyperSparseAutoencoder(d_model, hidden_features, feature_dim)
    out_BD, losses = sae.forward_train(x_BD)
    print(out_BD.shape, losses)

if __name__ == "__main__":
    sanity_check()
    test_decode()