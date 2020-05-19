import math
import tensorly as tl


def RSE(tensor, prediction):
    """
    Compute the RSE of the prediction.

    :param tensor: nd_array
    The true tensor.
    :param prediction: nd_array
    The predicted tensor.

    :return: float
    The RSE
    """

    diff = tensor - prediction
    return math.sqrt(tl.tenalg.inner(diff, diff) / tl.tenalg.inner(tensor, tensor))


def MSE(tensor, prediction):
    """
    Compute the MSE of the prediction.

    :param tensor: nd_array
    The true tensor.
    :param prediction: nd_array
    The predicted tensor.

    :return: float
    The MSE
    """

    diff = tensor - prediction
    return tl.tenalg.inner(diff, diff) / tensor.size

def PSNR(tensor, prediction):
    """
    Compute the peak signal-to-noise ratio (PSNR) of the prediction for RGB image data.

    :param tensor: nd_array
    The true tensor.
    :param prediction: nd_array
    The predicted tensor.

    :return: float
    The PSNR
    """

    return 10 * math.log10(255**2 / MSE(tensor, prediction))