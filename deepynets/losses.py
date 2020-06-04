import numpy as np
from scipy.special import xlogy
import inspect

class Loss:

    def __init__(self, fn, **kwargs):
        self.fn = fn
        self._fn_kwargs = kwargs

    def __call__(self, y_true, y_pred):
        return self.fn(y_true, y_pred, **self._fn_kwargs)

class BinaryCrossEntropy(Loss):

    def __init__(self, epsilon=1e-15):
        super(BinaryCrossEntropy, self).__init__(binary_cross_entropy, epsilon=epsilon)
        self.epsilon = epsilon

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    m = y_true.shape[0]
    return np.squeeze(-(1. / m) * np.nansum(np.multiply(y_true, np.log(y_pred + epsilon)) +
                                            np.multiply(1 - y_true, np.log(1 - y_pred + epsilon))))