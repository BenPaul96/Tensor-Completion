import numpy as np
import scipy.optimize as optimize
import tensorly as tl

from Code.Utils import f_unfold, f_fold, flatten_factors, unflatten_factors, build_W, TR_fold, TR_unfold, TR_to_tensor

# Use cupy if available, else numpy
use_cupy = False
try:
    import cupy as cp
    xp = cp
    use_cupy = True
except:
    xp = np


class TR_WOPT_Model(object):
    """
    This class implements the TR-WOPT algorithm.

    References
    ----------
    [1] Yuan, Cao, Zhao & Wu. (2017). Higher-dimension Tensor Completion via Low-rank Tensor Ring Decomposition.
    """

    def __init__(self, tensor, ranks, lr=None, optimization="gradient_descent"):
        """

        :param tensor: nd_array
        The tensor on which we want to execute the completion algorithm. The missing values should be np.nan.
        :param ranks: list of int
        The TR-ranks. They limit the size of every core tensors [1]. The first and last ranks must be equal.
        :param lr: float
        The learning rate for the optimization algorithm (None for ncg).
        :param optimization: "ncg" or "gradient_descent"
        The optimization algorithm used to optimize the factors matrices.
        """

        self.tensor = tensor
        self.ranks = ranks
        self.lr = lr
        self.optimization = optimization

        self.dims = tensor.shape
        self.n_dims = len(self.dims)
        self.factors = {}
        self.grads = {}
        self.shapes = [(self.ranks[i], self.dims[i], self.ranks[i + 1]) for i in range(self.n_dims)]
        self.W = build_W(self.tensor)
        self.Z = None
        self.train_logs = {}
        self.n_obs = self.W.sum()

        # So we don't have to compute them every time
        self.Y = xp.nan_to_num(self.tensor)
        self.gamma = tl.tenalg.inner(self.Y, self.Y)

        # Initialize the core tensors
        for mode in range(self.n_dims):
            self.factors[f"A{mode}"] = self.init_factors(mode)

    def init_factors(self, mode):
        """Initialize the factor matrix with a (0, 1) normal distribution"""
        return xp.random.normal(loc=0, scale=1, size=self.shapes[mode])

    def forward(self, factors=None):
        """Compute the objective function."""
        # Rearrange factors into a list of matrices.
        if self.optimization == "gradient_descent":
            factors = list(self.factors.values())
        elif self.optimization == "ncg":
            if use_cupy:
                factors = cp.asarray(factors)
            factors = unflatten_factors(factors, self.shapes)

        # Build the reconstruction of the tensor from the factors.
        self.Z = self.W * self.reconstruct(factors)

        # Compute the objective function.
        objective = 0.5 * self.gamma - tl.tenalg.inner(self.Y, self.Z) + 0.5 * tl.tenalg.inner(self.Z, self.Z)

        # If using cupy, objective will be a cupy array of 1 element. So we extract that element.
        if use_cupy:
            objective = objective.item()

        return objective

    def reconstruct(self, factors):
        """Reconstruct the tensor from the core tensors."""
        n = 1
        X_n = tl.dot(f_unfold(factors[n], 1), TR_unfold(TR_to_tensor(factors, n), 1).T)
        X = TR_fold(X_n, self.dims, n)

        return X

    def backward(self, factors=None):
        """Compute the gradients for each factor matrix."""
        if self.optimization == "gradient_descent":
            factors = list(self.factors.values())
        elif self.optimization == "ncg":
            if use_cupy:
                factors = cp.asarray(factors)
            factors = unflatten_factors(factors, self.shapes)

        # Compute the reconstruction of the tensor from the factors if it was not already done.
        if self.Z is None:
            self.Z = self.W * self.reconstruct(factors)

        # Compute the gradient for each factors matrix.
        T = self.Z - self.Y
        for n in range(self.n_dims):
            G = tl.dot(TR_unfold(T, n), TR_unfold(TR_to_tensor(factors, n), 1))
            G = f_fold(G, self.shapes[n], 1)
            self.grads[f"G{n}"] = G

        if self.optimization == "ncg":
            grads = flatten_factors(list(self.grads.values()))
            if use_cupy:
                grads = cp.asnumpy(grads)
            return grads

    def update(self):
        """Update the factors matrices."""
        for n in range(self.n_dims):
            self.factors[f"A{n}"] = xp.subtract(self.factors[f"A{n}"], self.lr * self.grads[f"G{n}"])

    def train(self, nb_epochs=5):
        """Train the model using the specified optimization algorithm."""
        if self.optimization == "gradient_descent":
            self.gradient_descent(nb_epochs)
        elif self.optimization == "ncg":
            self.ncg(nb_epochs)

    def gradient_descent(self, nb_epochs=5):
        """Use the gradient descent algorithm to train the model."""
        for epoch in range(nb_epochs):
            # We write the normalized loss to our train logs
            loss = self.forward() / self.n_obs
            self.train_logs[f"Epoch{epoch}"] = loss

            self.backward()
            self.update()
            print(f"Epoch: {epoch}, Loss: {loss}")

    def ncg(self, nb_epochs):
        """Use the Nonlinear Conjugate Gradient method to train the model."""
        x0 = list(self.factors.values())

        # Scipy takes in vectors, so we must flatten our factors to 1D, and then unflatten the optimized factors
        # back to the original shape. Also, x0 cannot be a cupy array because of scipy.
        x0 = flatten_factors(x0)
        if use_cupy:
            x0 = cp.asnumpy(x0)

        res = optimize.minimize(self.forward, x0, method="CG", jac=self.backward,
                                options={"disp": True, "maxiter": nb_epochs})
        factors = res.x

        if use_cupy:
            factors = cp.asarray(factors)
        factors = unflatten_factors(factors, self.shapes)

        self.train_logs = res.fun / self.n_obs
        for n in range(self.n_dims):
            self.factors[f"A{n}"] = factors[n]

    def predict(self, use_observed=True):
        """Compute the reconstruction of the tensor from the factors matrices."""
        factors = list(self.factors.values())
        reconstruction = self.reconstruct(factors)

        if use_observed:
            # We keep observed values, and fill the unobserved values with our predictions
            W_complement = xp.where((self.W == 0) | (self.W == 1), self.W ^ 1, self.W)
            prediction = W_complement * reconstruction + self.Y
        else:
            prediction = reconstruction
        return prediction



