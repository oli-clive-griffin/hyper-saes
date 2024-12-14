import torch as t

from hyper_sae import AutoEncoder, HyperSparseAutoencoder, SparseAutoencoder
from vis import visualize_weights_2d


class ModelTrainer:
    def __init__(self, name: str, model: AutoEncoder, optimizer: t.optim.Optimizer):
        self.name = name
        self.model = model
        self.optimizer = optimizer

    def train_step(self, x_BD: t.Tensor) -> float:
        self.optimizer.zero_grad()
        _, losses = self.model.forward_train(x_BD)
        loss = losses.reconstruction # + 0.02 * losses.sparsity
        loss.backward()
        self.optimizer.step()
        self.model.enforce_decoder_norm()
        return loss.item()


DIMENSIONALITY = 2


if __name__ == "__main__":
    batch_size = 2048
    num_iterations = 1_000_000
    log_interval = 1000
    lr = 3e-3

    regular_model = SparseAutoencoder(d_model=DIMENSIONALITY, hidden_features=2)
    hyper_regular_model = HyperSparseAutoencoder(d_model=DIMENSIONALITY, hidden_features=DIMENSIONALITY, feature_dim=1)
    hyper_model = HyperSparseAutoencoder(d_model=DIMENSIONALITY, hidden_features=1, feature_dim=DIMENSIONALITY)

    assert (
        regular_model.hidden_features
        == hyper_regular_model.hidden_features * hyper_regular_model.feature_dim
        == hyper_model.hidden_features * hyper_model.feature_dim
    )

    model_trainers = [
        ModelTrainer("regular", regular_model, t.optim.Adam(regular_model.parameters(), lr=lr)), 
        ModelTrainer("hyper_regular", hyper_regular_model, t.optim.Adam(hyper_regular_model.parameters(), lr=lr)), # should be the same as above
        ModelTrainer("hyper", hyper_model, t.optim.Adam(hyper_model.parameters(), lr=lr)), # should be the best
    ]

    def generate_clock() -> t.Tensor:
        """generates a 2d unit vector in a random direction"""
        rotation_N2 = t.randn(batch_size, DIMENSIONALITY)
        rotation_N2 = rotation_N2 / t.norm(rotation_N2, dim=1, keepdim=True)
        return rotation_N2

    for i in range(num_iterations):
        x_BD = generate_clock()
        losses = [model_trainer.train_step(x_BD) for model_trainer in model_trainers]
        if i % log_interval == 0:
            losses_str = ", ".join(
                f"{model_trainer.name}: {loss:.4f}" for loss, model_trainer in zip(losses, model_trainers)
            )
            print(f"Iteration {i}:\n{losses_str}")
            # for model_trainer in model_trainers:
            #     visualize_weights_2d(model_trainer.model.get_effective_weights())



# """
# Training on toy data with functional configs
# """

# from functools import partial
# import torch as t
# from einops import rearrange
# from vis import visualize_weights_3d, visualize_weights_2d
# from torch import nn
# from typing import Callable, Optional


# # def make_sparse_data(batch_size: int, model_dim: int, sparsity: float) -> t.Tensor:
# #     """Generate sparse binary data"""
# #     return (t.randn(batch_size, model_dim) < sparsity).float()


# # def make_antipodal_data(batch_size: int, model_dim: int) -> t.Tensor:
# #     """Generate antipodal data pairs"""
# #     base = t.zeros(batch_size, model_dim // 2, 2)
# #     which_pair_member = t.randint(0, 2, (batch_size, model_dim // 2)).long()
# #     x = base.clone()
# #     x[which_pair_member] = 1
# #     return rearrange(x, "b f1 f2 -> b (f1 f2)")


# # # Experiment configurations as partially applied functions
# # CONFIGS = {
# #     "easy": {
# #         "data_dim": 2,
# #         "hidden_dim": 2,
# #         "data_fn": partial(make_sparse_data, sparsity=1.0),
# #     },
# #     "pyramid": {
# #         "data_dim": 4,
# #         "hidden_dim": 3,
# #         "data_fn": partial(make_sparse_data, sparsity=0.04),
# #     },
# #     "3d_cross": {
# #         "data_dim": 6,
# #         "hidden_dim": 3,
# #         "data_fn": make_antipodal_data,
# #     },
# #     "cross": {
# #         "data_dim": 4,
# #         "hidden_dim": 2,
# #         "data_fn": partial(make_sparse_data, sparsity=0.01),
# #     },
# #     "pentagram": {
# #         "data_dim": 5,
# #         "hidden_dim": 2,
# #         "data_fn": partial(make_sparse_data, sparsity=0.01),
# #     },
# #     "hexagram": {
# #         "data_dim": 6,
# #         "hidden_dim": 2,
# #         "data_fn": partial(make_sparse_data, sparsity=0.01),
# #     },
# # }


# class AutoEncoder(nn.Module):
#     def __init__(self, model_dim: int, hidden_dim: int):
#         super().__init__()
#         self.W_in_FH = nn.Parameter(t.randn(model_dim, hidden_dim))
#         with t.no_grad():
#             norms_F = t.norm(self.W_in_FH, dim=1)
#             self.W_in_FH = self.W_in_FH / norms_F[:, None]

#         self.W_out_HF = nn.Parameter(t.randn(hidden_dim, model_dim))
#         with t.no_grad():
#             norms_H = t.norm(self.W_out_HF, dim=1)
#             self.W_out_HF = self.W_out_HF / norms_H[:, None]

#     def forward(self, x: t.Tensor) -> t.Tensor:
#         return self.linear_out(self.linear_in(x))

#     def forward_train(self, x: t.Tensor) -> t.Tensor:
#         out = self.forward(x)
#         return t.mean(t.square(out - x) * self.weights_F)


# class Trainer:
#     def __init__(self, model_dim: int, hidden_dim: int, learning_rate: float, data_fn: Callable[[int, int], t.Tensor]):
#         self.data_generator = data_fn(model_dim)
#         self.model = AutoEncoder(model_dim, hidden_dim)
#         self.optimizer = t.optim.Adam(self.model.parameters(), lr=learning_rate)
#         self.iteration = 0

#     def train_step(self):
#         self.optimizer.zero_grad()
#         x = self.data_generator.generate()
#         loss = self.model.forward_train(x)
#         loss.backward()
#         self.optimizer.step()
#         return loss

#     def train(self, num_iterations: Optional[int] = None):
#         while True if num_iterations is None else self.iteration < num_iterations:
#             loss = self.train_step()

#             if self.iteration % 10_000 == 0:
#                 lr = 0.01 * (0.99 ** (self.iteration // 500))
#                 print(f"Iteration {self.iteration}, lr: {lr:.6f}, loss: {loss:.6f}")

#             self.iteration += 1


# def main(experiment_name: str):
#     config = CONFIGS[experiment_name]
#     trainer = Trainer(config)
#     trainer.train()


# if __name__ == "__main__":
#     experiment_name = "3d_cross"
#     print(f"Training configuration: {experiment_name}")
#     main(experiment_name)
