import numpy as np


class Relu:

    def __call__(self, z):
        return np.maximum(z, 0)

    def derivative(self, z):
        return (z > 0).astype(z.dtype)


class Sigmoid:

    def __call__(self, z):
        return 1 / (1 + np.exp(-z))

    def derivative(self, z):
        return z * (1 - z)


class LeakyRelu:

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, z):
        return np.where(z > 0, z, z * self.alpha)

    def derivative(self, z):
        dz = np.ones_like(z)
        dz[z < 0] = self.alpha
        return dz


class Tanh:

    def __call__(self, z):
        return np.tanh(z)

    def derivative(self, z):
        return 1 - (self(z) ** 2)
