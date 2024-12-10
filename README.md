# N-SAEs

Sparse autoencoders can be thought of as extracting a 1 dimensional subspace in high-dimensional space
(and cleaning up interference with a ReLU). Why can't the same idea work for higher-dimensional features?

N-SAEs ($\text{sae}^n$??) attempt to do this. By allowing hidden features to be n > 1 dimensional, and
applying bias and relu in the 1d subspace of the feature's direction.
