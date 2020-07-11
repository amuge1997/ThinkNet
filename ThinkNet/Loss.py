from .Tool import checkShape
import numpy as n

class Loss:
    def __init__(self):
        pass

    def __call__(self, predY, realY):
        raise Exception('Loss is None')

    def loss(self, predY, realY):
        raise Exception('Loss is None')

    def grad(self, predY, realY):
        raise Exception('Loss is None')

class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    @checkShape
    def __call__(self, predY, realY):
        n_samples = predY.shape[0]
        Y = n.sum((realY - predY) ** 2) / n_samples
        return Y

    @checkShape
    def grad(self, predY, realY):
        n_samples = predY.shape[0]
        grad = (realY - predY) * (-1) / n_samples
        return grad

class NLLoss(Loss):
    def __init__(self):
        super().__init__()

    @checkShape
    def __call__(self, predY, realY):
        n_samples = predY.shape[0]

        a = n.exp(predY - n.max(predY, axis=1, keepdims=True))
        a = a / n.sum(a, axis=1, keepdims=True)
        C = realY * n.log(a)
        C = - n.sum( C,axis=1 )
        C = n.sum( C )
        Y = C / n_samples

        return Y

    @checkShape
    def grad(self, predY, realY):
        n_samples = predY.shape[0]
        grad = (n.copy(predY) - realY)
        grad = grad / n_samples
        return grad
















