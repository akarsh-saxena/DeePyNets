import numpy as np


class Zeros:

    def __call__(self, shape):
        return np.zeros(shape)


class Ones:

    def __call__(self, shape):
        return np.zeros(shape)


class RandomNormal:

    def __init__(self, mean=0.0, sd=1.0, scale=0.01):
        self.mean = mean
        self.sd = sd
        self.scale=scale

    def __call__(self, shape):
        return np.random.normal(loc=self.mean, scale=self.sd, size=shape)
