import numpy as np


class Accuracy:

    def __call__(self, y_true, y_pred):
        return calculate_accuracy(y_true, y_pred)


def calculate_accuracy(y_true, y_pred):

    if y_true.shape[1] == 1:
        acc = (np.where(y_pred > 0.5, 1, 0) == y_true).sum() / len(y_true)
    else:
        acc = (np.argmax(y_true, 1) == np.argmax(y_pred, 1)).sum() / len(y_true)
    return acc
    