import numpy as np
import tensorly as tl

def f_unfold(tensor, mode=0):
    """
    Simple unfolding function.
    Needed because unfolding in tensorly follows C order, which is not the same as the one used in
    the Kolda Bader paper.
        Moves the `mode` axis to the beginning and reshapes in Fortran order
        source: http://jeankossaifi.com/blog/unfolding.html
    """
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='F')


def flatten_factors(factors):
    """
    Flatten a list of tensors into a 1D array.

    :param factors: list of nd_array
    The factors tensors to flatten

    :return: 1D nd_array
    A 1D array containing the flattened factors tensors.
    """

    return np.concatenate([factors[i].flatten() for i in range(len(factors))])


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
    mask = np.random.choice(img.size//3, n_missing_pixels, replace=False) * 3

    img = img.astype(np.float64)
    for i in range(3):
        img.ravel()[mask + i] = np.nan

    return img

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
        return tl.tensor([1])

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

    return np.squeeze(tl.reshape(full_tensor, full_shape))
