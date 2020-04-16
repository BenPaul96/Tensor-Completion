import tensorly as tl
import numpy as np


# Two old functions of TT_WOPT. Replaced by TT_to_tensor
def left_partial_construct(self, n):
    """Compute G(<n) as seen in Yuan, Zhao, Cao [1]"""
    if n == 0:
        return tl.tensor([1])

    left_factors = list(self.factors.values())[:n]

    # Derived from tensorly mps_to_tensor implementation
    # full_shape is the shape of the tensor to return
    full_shape = [f.shape[1] for f in left_factors]
    full_shape.append(left_factors[-1].shape[2])

    # We iterate over the left tensors. We accumulate the results in full_tensor
    full_tensor = np.reshape(left_factors[0], (full_shape[0], -1), order="F")
    for factor in left_factors[1:]:
        rank_prev, _, rank_next = factor.shape
        factor = np.reshape(factor, (rank_prev, -1), order="F")
        full_tensor = tl.dot(full_tensor, factor)
        full_tensor = np.reshape(full_tensor, (-1, rank_next), order="F")

    return np.reshape(full_tensor, full_shape, order="F")


def right_partial_construct(self, n):
    """Compute G(>n) as seen in Yuan, Zhao, Cao [1]"""
    if n == self.num_dims - 1:
        return tl.tensor([1])

    right_factors = list(self.factors.values())[n + 1:]

    # Derived from tensorly mps_to_tensor implementation
    # full_shape is the shape of the tensor to return
    full_shape = [f.shape[1] for f in right_factors]
    full_shape.insert(0, right_factors[0].shape[0])

    # We iterate over the right tensors. We accumulate the results in full_tensor
    full_tensor = np.reshape(right_factors[0], (full_shape[0] * full_shape[1], -1), order="F")
    for factor in right_factors[1:]:
        rank_prev, _, rank_next = factor.shape
        factor = np.reshape(factor, (rank_prev, -1), order="F")
        full_tensor = tl.dot(full_tensor, factor)
        full_tensor = np.reshape(full_tensor, (-1, rank_next), order="F")

    return np.reshape(full_tensor, full_shape, order="F")