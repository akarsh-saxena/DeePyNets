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


class GlorotUniform:

    def __init__(self, constant=6.0, random_state=None):
        self.constant = constant
        self.random_state = random_state

    def __call__(self, shape):
        np.random.seed(self.random_state)
        sd = np.sqrt(self.constant / (shape[0] + shape[1]))
        ar = np.random.uniform(-sd, sd, (shape[0], shape[1]))
        return ar


aliases = {
    'zeros': Zeros(),
    'ones': Ones(),
    'glorot_uniform': GlorotUniform(),
    'random_normal': RandomNormal()
}


def get(initializer):
    if isinstance(initializer, str):
        return aliases[initializer]
    elif callable(initializer):
        return initializer
    else:
        raise ValueError('Parameter type not understood')

