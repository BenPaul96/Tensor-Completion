import numpy as np
import tensorly as tl
import math


class Model(object):
    def __init__(self, data=None):
        if data:
            self.data = data

    def predict(self, method="TR-WOPT"):
        if method == "TR-WOPT":
            pass
        elif method == "TT_WOPT":
            pass
        elif method == "CP-WOPT":
            return self.CP_WOPT(self.data)
        else:
            raise Exception("Unknown tensor completion method")

    def CP_WOPT(self, tensor):
        return None

    def evaluate(self, prediction, metric="RSE"):
        if metric == "RSE":
            return self.RSE(prediction)
        elif metric == "PSRE":
            pass
        else:
            raise Exception("Unkown evaluation metric")

    def RSE(self, prediction):
        diff = self.data - prediction
        return (tl.tenalg.inner(diff, diff) / tl.tenalg.inner(self.data, self.data)) ** (1 / 2)

    def set_data(self, data):
        self.data = data

