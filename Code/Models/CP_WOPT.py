import numpy as np
import scipy.optimize as optimize
import tensorly as tl

from Code.Utils import f_unfold, flatten_factors, unflatten_factors, build_W

# Use cupy if available, else numpy
use_cupy = False
try:
    import cupy as cp
    xp = cp
    use_cupy = True
except:
    xp = np


class CP_WOPT_Model(object):
    """This class implements the CP-WOPT algorithm"""

    def __init__(self, tensor, rank, lr=None, init="SVD", optimization="gradient_descent", seed=None):
        """
        :param tensor: nd_array
        The tensor on which we want to execute the completion algorithm. The missing values should be np.nan.
        :param rank: int
        The assumed rank of <tensor>. Also the second dimension of the factors matrices.
        :param lr: float
        The learning rate for the optimization algorithm (None for ncg).
        :param init: "normal" or "SVD"
        The initialization method used to initialize the factors matrices.
        :param optimization: "ncg" or "gradient_descent"
        The optimization algorithm used to optimize the factors matrices.
        :param seed: int
        If given, set the seed to obtain reproducible results.
        """

        self.tensor = tensor
        self.rank = rank
        self.lr = lr
        self.initialization = init
        self.optimization = optimization

        self.dims = tensor.shape
        self.n_dims = len(self.dims)
        self.factors = {}
        self.grads = {}
        self.shapes = [(self.dims[i], self.rank) for i in range(self.n_dims)]  # The shapes of the factors tensors.
        self.W = build_W(self.tensor)
        self.Z = None
        self.train_logs = {}
        self.n_obs = self.W.sum()
        if seed is not None:
            xp.random.seed(seed)

        # So we don't have to compute them every time
        self.Y = xp.nan_to_num(self.tensor)
        self.gamma = tl.tenalg.inner(self.Y, self.Y)

        for n in range(self.n_dims):
            self.factors[f"A{n}"] = self.init_factors(n, self.initialization)

    def init_factors(self, mode, initialization):
        """
        Call the appropriate initialization method for a factor matrix.

        :param mode: int
        The mode (dimension) represented by the factor matrix.
        :param initialization: "normal" or "SVD"
        The initialization method.
        :return nd_array
        The initialized factor matrix.
        """

        if initialization == "normal":
            return self.normal_init(mode)
        elif initialization == "SVD":
            return self.SVD_init(mode)

    def normal_init(self, mode):
        """Initialize the factor matrix with a (0, 1) normal distribution"""
        return xp.random.normal(loc=0, scale=1, size=self.shapes[mode])

    def SVD_init(self, mode):
        """Initialize the factor matrix using the n-mode singular vectors"""
        # TODO make compatible with cupy
        U, _, _ = tl.partial_svd(f_unfold(self.Y, mode), n_eigenvecs=self.rank)

        # Fill U from normal distribution if the dimension of the present mode is smaller than the rank.
        if self.Y.shape[mode] < self.rank:
            random_part = xp.random.normal(loc=0, scale=1, size=(self.dims[mode], self.rank - self.Y.shape[mode]))
            U = tl.concatenate([U, random_part], axis=1)

        return U

    def forward(self, factors=None):
        """Compute the objective function"""
        # Rearrange factors into a list of matrices.
        if self.optimization == "gradient_descent":
            factors = list(self.factors.values())
        elif self.optimization == "ncg":
            if use_cupy:
                factors = cp.asarray(factors)
            factors = unflatten_factors(factors, self.shapes)

        # Build the reconstruction of the tensor from the factors.
        reconstruction = tl.kruskal_tensor.kruskal_to_tensor((None, factors))
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
        # Rearrange factors into a list of matrices
        if self.optimization == "gradient_descent":
            factors = list(self.factors.values())
        elif self.optimization == "ncg":
            if use_cupy:
                factors = cp.asarray(factors)
            factors = unflatten_factors(factors, self.shapes)

        # Compute the reconstruction of the tensor from the factors if it was not already done.
        if self.Z is None:
            reconstruction = tl.kruskal_tensor.kruskal_to_tensor((None, factors))
            if use_cupy:
                reconstruction = cp.asarray(reconstruction)
            self.Z = self.W * reconstruction

        # Compute the gradient for each factors matrix.
        T = self.Y - self.Z
        for n in range(self.n_dims):
            An_neg = tl.tenalg.khatri_rao(factors, skip_matrix=n, reverse=True)
            self.grads[f"G{n}"] = tl.dot(f_unfold(T, n), An_neg) * -1

        # Flatten the gradients for ncg.
        if self.optimization == "ncg":
            grads = flatten_factors(list(self.grads.values()))
            if use_cupy:
                grads = cp.asnumpy(grads)
            return grads

    def update(self):
        """Update the factors matrices."""
        for n in range(self.n_dims):
            self.factors[f"A{n}"] = xp.subtract(self.factors[f"A{n}"], self.lr * self.grads[f"G{n}"])

    def train(self, n_epochs=5):
        """Train the model using the specified optimization algorithm."""
        if self.optimization == "gradient_descent":
            self.gradient_descent(n_epochs)
        elif self.optimization == "ncg":
            self.ncg(n_epochs)

    def gradient_descent(self, n_epochs=5):
        """Use the gradient descent algorithm to train the model."""
        for epoch in range(n_epochs):
            # We write the normalized loss to our train logs
            loss = self.forward() / self.n_obs
            self.train_logs[f"Epoch{epoch}"] = loss

            self.backward()
            self.update()
            print(f"Epoch: {epoch}, Loss: {loss}")

    def ncg(self, n_epochs):
        """Use the Nonlinear Conjugate Gradient method to train the model."""
        x0 = list(self.factors.values())

        # Scipy takes in vectors, so we must flatten our factors to 1D, and then unflatten the optimized factors
        # back to the original shape. Also, x0 cannot be a cupy array because of scipy.
        x0 = flatten_factors(x0)
        if use_cupy:
            x0 = cp.asnumpy(x0)

        res = optimize.minimize(self.forward, x0, method="CG", jac=self.backward,
                                options={"disp": True, "maxiter": n_epochs})
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
        reconstruction = tl.kruskal_tensor.kruskal_to_tensor((None, list(self.factors.values())))

        if use_observed:
            # We keep observed values, and fill the unobserved values with our predictions
            W_complement = xp.where((self.W == 0) | (self.W == 1), self.W ^ 1, self.W)
            prediction = W_complement * reconstruction + self.Y
        else:
            prediction = reconstruction

        return prediction
