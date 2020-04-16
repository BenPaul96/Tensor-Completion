import numpy as np
import tensorly as tl
from Code.Utils import f_unfold, flatten_factors, unflatten_factors, TT_to_tensor
import scipy.optimize as optimize


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
        The TT-ranks. They limit the size of every core tensors. Must start and end by 1.
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

        self.dims = tensor.shape
        self.n_dims = len(self.dims)
        self.factors = {}
        self.grads = {}
        self.shapes = [(self.ranks[i], self.dims[i], self.ranks[i + 1]) for i in range(self.n_dims)]
        self.W = self.build_W()
        self.Z = None
        self.train_logs = {}
        self.optimization = optimization
        self.n_obs = self.W.sum()
        if seed is not None:
            np.random.seed(seed)

        # So we don't have to compute them every time
        self.Y = np.nan_to_num(self.tensor)
        self.gamma = tl.tenalg.inner(self.Y, self.Y)

        for mode in range(self.n_dims):
            self.factors[f"A{mode}"] = self.init_factors(mode)

    def build_W(self):
        """
        Build the weights tensor.

        :return W: nd_array
        A tensor of the same shape as <self.tensor>, with ones at the position of known entries and zeros elsewhere.
        """

        W = self.tensor.copy()
        W[~np.isnan(W)] = 1
        W = np.nan_to_num(W)

        return W

    def init_factors(self, mode):
        """Initialize the factor matrix with a (0, 1) normal distribution"""
        return np.random.normal(loc=0, scale=1, size=self.shapes[mode])

    def forward(self, factors=None):
        """Compute the objective function."""
        # Rearrange factors into a list of matrices.
        if self.optimization == "gradient_descent":
            factors = list(self.factors.values())
        elif self.optimization == "ncg":
            factors = unflatten_factors(factors, self.shapes)

        # Build the reconstruction of the tensor from the factors.
        self.Z = self.W * tl.mps_tensor.mps_to_tensor(factors)
        # Compute the objective function.
        return 0.5 * self.gamma - tl.tenalg.inner(self.Y, self.Z) + 0.5 * tl.tenalg.inner(self.Z, self.Z)

    def backward(self, factors=None):
        """Compute the gradients for each factor matrix."""
        if self.optimization == "gradient_descent":
            factors = list(self.factors.values())
        elif self.optimization == "ncg":
            factors = unflatten_factors(factors, self.shapes)

        # Compute the reconstruction of the tensor from the factors if it was not already done.
        if self.Z is None:
            self.Z = self.W * tl.mps_to_tensor(factors)

        # Compute the gradient for each factors matrix.
        T = self.Z - self.Y
        for n in range(self.n_dims):
            # Note that we inverse the order of the kronecker product compared to [1]. This is because all tensor
            # operations are done in C order in tensorly instead of F order, so we have to adjust the kronecker product.
            X = tl.tenalg.kronecker(
                [tl.unfold(TT_to_tensor(factors, 0, n), n), tl.unfold(TT_to_tensor(factors, n+1, self.n_dims), 0)])

            G = tl.dot(tl.base.unfold(T, n), X.T)
            G = tl.base.fold(G, 1, (self.ranks[n], self.dims[n], self.ranks[n + 1]))

            self.grads[f"G{n}"] = G

        if self.optimization == "ncg":
            return flatten_factors(list(self.grads.values()))

    def update(self):
        """Update the factors matrices."""
        for n in range(self.n_dims):
            self.factors[f"A{n}"] = np.subtract(self.factors[f"A{n}"], self.lr * self.grads[f"G{n}"])

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
        x0 = flatten_factors(x0)

        res = optimize.minimize(self.forward, x0, method="CG", jac=self.backward,
                                options={"disp": True, "maxiter": nb_epochs})

        factors = unflatten_factors(res.x, self.shapes)
        self.train_logs = res.fun / self.n_obs
        for n in range(self.n_dims):
            self.factors[f"A{n}"] = factors[n]

    def predict(self):
        """Compute the reconstruction of the tensor from the factors matrices."""
        return tl.mps_tensor.mps_to_tensor(list(self.factors.values()))
