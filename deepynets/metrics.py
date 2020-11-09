import numpy as np


class Accuracy:

    def __call__(self, y_true, y_pred):
        return calculate_accuracy(y_true, y_pred)


class Precision:

    def __call__(self, y_true, y_pred):
        return calculate_precision(y_true, y_pred)


class Recall:

    def __call__(self, y_true, y_pred):
        return calculate_recall(y_true, y_pred)


def confusion_matrix(y_true, y_pred):

    y_pred = np.where(y_pred > 0.5, 1, 0)

    if y_true.shape[1] == 1:
        n_classes = 2
        y_true = np.squeeze(np.eye(n_classes)[y_true.reshape(-1)])
        y_pred = np.squeeze(np.eye(n_classes)[y_pred.reshape(-1)])
    else:
        n_classes = y_true.shape[1]

    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, [y_true.argmax(1), y_pred.argmax(1)], 1)

    return cm


def calculate_accuracy(y_true, y_pred):

    if y_true.shape[1] == 1:
        acc = (np.where(y_pred > 0.5, 1, 0) == y_true).sum() / len(y_true)
    else:
        acc = (np.argmax(y_true, 1) == np.argmax(y_pred, 1)).sum() / len(y_true)
    return acc
    

def calculate_precision(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    num = np.diag(cm)
    den = np.sum(cm, axis = 0)

    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den!=0)


def calculate_recall(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    num = np.diag(cm)
    den = np.sum(cm, axis = 1)

    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den!=0)
