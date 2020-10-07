from .cTools import checkShape
import numpy as n


class Loss:
    def __init__(self):
        pass

    def __call__(self, predict_y, real_y):
        raise Exception('Loss is None')

    def loss(self, predict_y, real_y):
        raise Exception('Loss is None')

    def grad(self, predict_y, real_y):
        raise Exception('Loss is None')


class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    @checkShape
    def __call__(self, predict_y, real_y):
        n_samples = predict_y.shape[0]
        Y = n.sum((real_y - predict_y) ** 2) / n_samples
        return Y

    @checkShape
    def grad(self, predict_y, real_y):
        n_samples = predict_y.shape[0]
        grad = (real_y - predict_y) * (-1) / n_samples
        return grad


class NLLoss(Loss):
    def __init__(self):
        super().__init__()

    @checkShape
    def __call__(self, predict_y, real_y):
        n_samples = predict_y.shape[0]

        a = n.exp(predict_y - n.max(predict_y, axis=1, keepdims=True))
        a = a / n.sum(a, axis=1, keepdims=True)
        c = real_y * n.log(a)
        c = - n.sum(c, axis=1)
        c = n.sum(c)
        y = c / n_samples

        return y

    @checkShape
    def grad(self, predict_y, real_y):
        n_samples = predict_y.shape[0]
        grad = (n.copy(predict_y) - real_y)
        grad = grad / n_samples
        return grad
















