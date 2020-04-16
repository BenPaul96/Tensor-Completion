import cv2
import numpy as np
import tensorly as tl

from Code.Models.CP_WOPT import CP_WOPT_Model
from Code.Models.TT_WOPT import TT_WOPT_Model
from Code.Utils import mask_img, TT_to_tensor


# Read lena image
img = cv2.imread('../data/lena.jpg', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (200, 200))

# Mask image
img_missing = mask_img(img)

def test_CP_training():
    model = model = CP_WOPT_Model(img_missing, 24, 1e-6, init="SVD", optimization="ncg", seed=0)
    model.train(5000)

def test_TT_training():
    model = TT_WOPT_Model(img_missing, (1, 24, 24, 1), optimization="ncg")
    model.train(5000)

def test_tensorly_kronecker():
    A = np.array([[1,2], [3,4]])
    B = np.array([[5,6], [7,8]])

    print(tl.tenalg.kronecker([A, B]))

def test_tensorly_khatri_rao():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    print(tl.tenalg.khatri_rao([A, B]))


def test_TT_to_tensor():
    # Tests each slice for n in range(3)
    A1 = np.arange(1, 7).reshape((1, 3, 2))
    A2 = np.arange(12, 0, -1).reshape((2, 3, 2))
    A3 = np.arange(1, 5).reshape((2, 2, 1))
    factors = [A1, A2, A3]

    # Sanity test by comparing to tensorly's mps_to_tensor
    assert np.allclose(tl.mps_tensor.mps_to_tensor(factors), TT_to_tensor(factors, 0, 3))

    # Test (0, 0)
    true_val = [1]
    assert TT_to_tensor(factors, 0, 0) == true_val

    # Test (0, 1)
    full_shape = (3, 2)
    true_val = A1.reshape(full_shape)
    assert np.array_equal(true_val, TT_to_tensor(factors, 0, 1))

    # Test (0, 2)
    full_shape = (3, 3, 2)
    true_val = np.dot(A1.reshape((3, 2)), A2.reshape((2, 6))).reshape(full_shape)
    assert np.allclose(true_val, TT_to_tensor(factors, 0, 2))

    # Test (0, 3)
    full_shape = (3, 3, 2)
    true_val = np.dot(true_val.reshape(9, 2), A3.reshape(2, 2)).reshape(full_shape)
    assert np.allclose(true_val, TT_to_tensor(factors, 0, 3))

    # Test (1, 1)
    true_val = [1]
    assert TT_to_tensor(factors, 1, 1) == true_val

    # Test (1, 2)
    true_val = A2
    assert np.array_equal(true_val, TT_to_tensor(factors, 1, 2))

    # Test (1, 3)
    full_shape = (2, 3, 2)
    true_val = np.dot(A2.reshape((6, 2)), A3.reshape((2, 2))).reshape(full_shape)
    assert np.allclose(true_val, TT_to_tensor(factors, 1, 3))

    # Test (2, 2)
    true_val = [1]
    assert TT_to_tensor(factors, 2, 2) == true_val

    # Test (2, 3)
    full_shape = (2, 2)
    true_val = A3.reshape(full_shape)
    assert np.array_equal(true_val, TT_to_tensor(factors, 2, 3))

    # Test (3, 3)
    true_val = [1]
    assert TT_to_tensor(factors, 3, 3) == true_val


def test_TT_reconstruction():
    '''Test that X_(n) = G^(n)_(2) (G^(>n)_(1) x G^(<n)_(n)) where x is the kronecker product'''
    model = TT_WOPT_Model(img_missing, (1, 24, 24, 1))
    factors = list(model.factors.values())
    X = tl.mps_to_tensor(factors)
    n_dims = len(factors)

    for n in range(n_dims):
        print(n)
        X_n = tl.unfold(X, n)
        G_n = tl.unfold(factors[n], 1)
        Y = tl.tenalg.kronecker(
            [tl.unfold(TT_to_tensor(factors, 0, n), n), tl.unfold(TT_to_tensor(factors, n+1, n_dims), 0)])

        result = tl.dot(G_n, Y)
        result = tl.fold(result, n, X.shape)
        assert np.allclose(X, result)


def test_reconstruction_analytic():
    '''Analytic test for TT reconstruction'''
    # Define the core tensors
    G1 = np.arange(1, 9).reshape((1, 2, 4))
    G2 = np.arange(1, 25).reshape((4, 3, 2))
    G3 = np.arange(1, 9).reshape((2, 4, 1))
    factors = [G1, G2, G3]
    X = tl.mps_to_tensor(factors)
    n_dims = 3

    # Test n=1
    n=1
    G_2 = tl.unfold(G2, 1)
    G_gt_n = tl.unfold(TT_to_tensor(factors, n+1, n_dims), 0)
    G_lt_n = tl.unfold(TT_to_tensor(factors, 0, n), n)

    # Since we do all operations in C order instead of F, we have to reverse the kronecker product
    Y = tl.tenalg.kronecker([G_lt_n, G_gt_n])
    result = tl.dot(G_2, Y)
    assert np.allclose(tl.unfold(X, n), result)



test_TT_reconstruction()

