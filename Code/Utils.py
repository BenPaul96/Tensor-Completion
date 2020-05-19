import numpy as np
import tensorly as tl

# Use cupy if available, else numpy
use_cupy = False
try:
    import cupy as cp
    xp = cp
    use_cupy = True
except:
    xp = np

def f_unfold(tensor, mode=0):
    """
    Simple unfolding function.
    Needed because unfolding in tensorly follows C order, which is not the same as the one used in
    the Kolda Bader paper.
        Moves the `mode` axis to the beginning and reshapes in Fortran order
        source: http://jeankossaifi.com/blog/unfolding.html
    """
    return xp.reshape(xp.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')

def f_fold(tensor, dims, mode=0):
    """
    F order folding function

    :param tensor: nd_array
    The unfolded tensor.
    :param dims: list of ints
    The dimensions of the original tensor.
    :param mode: int
    """

    n_dims = len(dims)
    order = np.concatenate(([mode], np.arange(mode), np.arange(mode+1, n_dims)))

    dims_ordered = [dim for i, dim in enumerate(dims) if i != mode]
    dims_ordered.insert(0, dims[mode])

    return xp.moveaxis(xp.reshape(tensor, dims_ordered, order="F"), np.arange(n_dims), order)

def TR_unfold(tensor, mode=0):
    """
    Unfolding function needed for TR-WOPT. It unfolds the tensor in the following order: (n x n+1, ..., N, 1, ..., n-1)
    :param tensor: nd_array
    :param mode: int
    """

    n_dims = len(tensor.shape)
    dims_order = np.concatenate((np.arange(mode, n_dims), np.arange(mode)))

    return xp.reshape(xp.moveaxis(tensor, dims_order, np.arange(n_dims)), (tensor.shape[mode], -1), order="F")

def TR_fold(tensor, dims, mode=0):
    """
    Folding function needed for TR-WOPT. Fold a tensor unfolded in the following order: (n x n+1, ..., N, 1, ..., n-1)

    :param tensor: nd_array
    The unfolded tensor.
    :param dims: list of ints
    The dimensions of the original tensor.
    :param mode: int
    """

    n_dims = len(dims)
    order = np.concatenate((np.arange(mode, n_dims), np.arange(mode)))
    dims_ordered = np.concatenate((dims[mode:], dims[:mode])).astype(int)

    return xp.moveaxis(xp.reshape(tensor, dims_ordered, order="F"), np.arange(n_dims), order)


def build_W(tensor):
    """
    Build the weights tensor, where each entry is one if the corresponding entry in the original tensor is observed,
    and 0 if it is not (unobserved entries are set to np.nan in the original tensor).

    :param tensor: nd_array
    The original tensor
    :return W: nd_array
    A tensor of the same shape as <self.tensor>, with ones at the position of known entries and zeros elsewhere.
    """

    W = tensor.copy()
    W[~xp.isnan(W)] = 1
    W = xp.nan_to_num(W)

    return W.astype(int)


def flatten_factors(factors):
    """
    Flatten a list of tensors into a 1D array.

    :param factors: list of nd_array
    The factors tensors to flatten

    :return: 1D nd_array
    A 1D array containing the flattened factors tensors.
    """

    return xp.concatenate([factors[i].flatten() for i in range(len(factors))])


def unflatten_factors(flattened_factors, shapes):
    """
    Unflatten a 1D array into a list of n_dims matrices where
    the nth matrix is of shape (dims[n], rank).

    :param flattened_factors: 1D nd_array
    The flatten factors.
    :param shapes: list of tuples of int
    The shape of each factors tensor.

    :return factors: list of 2D nd_array
    """

    factors = []
    for shape in shapes:
        n_elems = np.prod(shape)
        elems = flattened_factors[:n_elems]
        flattened_factors = flattened_factors[n_elems:]
        factors.append(elems.reshape(shape))

    return factors


def mask_img(img, missing_rate=0.5):
    """
    Randomly mask some pixels of an image.

    :param img: 3D nd_array
    :param missing_rate: float
    The fraction of pixels to remove from the image.

    :return: 3D nd_array
    The masked image
    """

    n_missing_pixels = int(missing_rate * img.size/3)
    mask = xp.random.choice(img.size//3, n_missing_pixels, replace=False) * 3

    img = img.astype(np.float64)
    for i in range(3):
        img.ravel()[mask + i] = np.nan

    return img


def mask_tensor(tensor, missing_rate=0.5):
    """
    Randomly mask a fraction of the elements of a tensor.

    :param tensor: nd_array
    :param missing_rate: float
    The fraction of pixels to remove from the image.

    :return: nd_array
    The masked tensor.
    """

    tensor_copy = tensor.copy()

    n_missing_pixels = int(missing_rate * tensor.size)
    mask = xp.random.choice(tensor.size, n_missing_pixels, replace=False)

    tensor_copy.ravel()[mask] = np.nan

    return tensor_copy

def TT_to_tensor(factors, start=0, end=None):
    """
    Reconstruct a tensor from the core tensors. Allows us to take only a slice of the core tensors
    We use this function to compute G(>n) and G(<n) as seen in Yuan, Zhao, Cao [1].

    :param factors: list of nd_array
    The core tensors.
    :param start: int
    The index of the first core tensor to use.
    :param end: int
    The index of the last core tensor to use.
    """

    # Take care of the edge cases G(<0) and G(> <n_dims>)
    if start == end:
        return xp.array([1])

    factors = factors[start:end]

    # Derived from tensorly mps_to_tensor implementation
    # full_shape is the shape of the tensor to return
    full_shape = [f.shape[1] for f in factors]
    full_shape.insert(0, factors[0].shape[0])
    full_shape.append(factors[-1].shape[2])

    # We iterate over the right tensors. We accumulate the results in full_tensor
    full_tensor = tl.reshape(factors[0], (full_shape[0] * full_shape[1], -1))
    for factor in factors[1:]:
        rank_prev, _, rank_next = factor.shape
        factor = tl.reshape(factor, (rank_prev, -1))
        full_tensor = tl.dot(full_tensor, factor)
        full_tensor = tl.reshape(full_tensor, (-1, rank_next))

    return xp.squeeze(tl.reshape(full_tensor, full_shape))

def TR_to_tensor(factors, n):
    """
    Computes G(!=n) [1].

    :param factors: list of nd_array
    The core tensors.
    :param n: int

    References
    ----------
    [1] Yuan, Cao, Zhao & Wu. (2017). Higher-dimension Tensor Completion via Low-rank Tensor Ring Decomposition.
    """

    # Edge cases for first and last mode
    if n == 0:
        rank_first = factors[1].shape[0]
        rank_last = factors[-1].shape[2]
        result = TT_to_tensor(factors[1:]).reshape((rank_first, -1, rank_last), order="F")
        return result
    elif n == len(factors) - 1:
        rank_first = factors[0].shape[0]
        rank_last = factors[n-1].shape[2]
        result = TT_to_tensor(factors[:-1]).reshape((rank_first, -1, rank_last), order="F")
        return result

    rank_first = factors[n+1].shape[0]
    rank_last = factors[n-1].shape[2]

    right_factors = factors[n+1:]
    left_factors = factors[:n]

    dims_product = np.array([f.shape[1] for f in right_factors + left_factors]).prod()
    full_shape = [dims_product]

    full_shape.insert(0, rank_first)
    full_shape.append(rank_last)

    A = TT_to_tensor(right_factors)
    B = TT_to_tensor(left_factors)

    A = xp.reshape(A, (-1, A.shape[-1]), order="F")
    B = xp.reshape(B, (B.shape[0], -1), order="F")

    result = tl.dot(A, B).reshape(full_shape, order="F")
    return result



