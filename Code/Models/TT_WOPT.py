import numpy as np
import tensorly as tl
import time

from Code.Utils import f_unfold, flatten_factors, unflatten_factors, TT_to_tensor, build_W
import scipy.optimize as optimize

# Use cupy if available, else numpy
use_cupy = False
try:
    import cupy as cp
    xp = cp
    use_cupy = True
except:
    xp = np


class TT_WOPT_Model(object):
    """
    This class implements the TT-WOPT algorithm.

    References
    ----------
    [1] Yuan, Longhao & Zhao, Qibin. (2017). Completion of High Order Tensor Data with Missing
    Entries via Tensor-train Decomposition.
    """

    def __init__(self, tensor, ranks, lr=None, optimization="gradient_descent", seed=None):
        """
        :param tensor: nd_array
        The tensor on which we want to execute the completion algorithm. The missing values should be np.nan.
        :param ranks: list of int
        The TT-ranks. They limit the size of every core tensors [1]. Must start and end by 1.
        :param lr: float
        The learning rate for the optimization algorithm (None for ncg).
        :param optimization: "ncg" or "gradient_descent"
        The optimization algorithm used to optimize the factors matrices.
        :param seed: int
        If given, set the seed to obtain reproducible results.
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
        if seed is not None:
            xp.random.seed(seed)

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
        reconstruction = TT_to_tensor(factors)
        if use_cupy:
            reconstruction = cp.asarray(reconstruction)
        self.Z = self.W * reconstruction

        # Compute the objective function.
        objective = 0.5 * self.gamma - tl.tenalg.inner(self.Y, self.Z) + 0.5 * tl.tenalg.inner(self.Z, self.Z)

        # If using cupy, objective will be a cupy array of 1 element. So we extract that element.
        if use_cupy:
            objective = objective.item()

        return objective

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
            reconstruction = TT_to_tensor(factors)
            if use_cupy:
                reconstruction = cp.asarray(reconstruction)
            self.Z = self.W * reconstruction

        # Compute the gradient for each factors matrix.
        T = self.Z - self.Y
        for n in range(self.n_dims):
            # Note that we inverse the order of the kronecker product compared to [1]. This is because all tensor
            # operations are done in C order in tensorly instead of F order, so we have to adjust the kronecker product.
            a = tl.unfold(TT_to_tensor(factors, 0, n), n)
            b = tl.unfold(TT_to_tensor(factors, n+1, self.n_dims), 0)

            if n == 0:
                X = b
            elif n == self.n_dims - 1:
                X = a
            else:
                X = xp.kron(a, b)

            G = tl.dot(tl.unfold(T, n), X.T)
            G = tl.fold(G, 1, (self.ranks[n], self.dims[n], self.ranks[n + 1]))

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
        # back to the original shape
        x0 = flatten_factors(x0)
        if use_cupy:
            x0 = cp.asnumpy(x0)

        res = optimize.minimize(self.forward, x0, method="CG", jac=self.backward,
                                options={"disp": True, "maxiter": nb_epochs})
        factors = res.x

        if use_cupy:
            factors = cp.asarray(factors)
        factors = unflatten_factors(factors, self.shapes)

        # We write the normalized loss the our train logs
        self.train_logs = res.fun / self.n_obs
        for n in range(self.n_dims):
            self.factors[f"A{n}"] = factors[n]

    def predict(self, use_observed=True):
        """Compute the reconstruction of the tensor from the factors matrices."""
        reconstruction = TT_to_tensor(list(self.factors.values()))

        if use_observed:
            # We keep observed values, and fill the unobserved values with our predictions
            W_complement = xp.where((self.W == 0) | (self.W == 1), self.W^1, self.W)
            prediction = W_complement * reconstruction + self.Y
        else:
            prediction = reconstruction

        return prediction
