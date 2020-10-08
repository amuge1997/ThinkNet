import numpy as n


def check_shape(infunc):
    def outfunc(ins, X1, X2):
        if X1.shape != X2.shape:
            raise Exception('数组形式不一致')
        else:
            ret = infunc(ins, X1, X2)
            return ret
    return outfunc


class Loss:
    def __init__(self):
        pass

    def __call__(self, prds, labs):
        raise Exception('Loss is None')

    def loss(self, predict_y, real_y):
        raise Exception('Loss is None')

    def grad(self, predict_y, real_y):
        raise Exception('Loss is None')


class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    @check_shape
    def __call__(self, prds, labs):
        n_samples = prds.shape[0]
        Y = n.sum((labs - prds) ** 2) / n_samples
        return Y

    @check_shape
    def grad(self, prds, labs):
        n_samples = prds.shape[0]
        grad = (labs - prds) * (-1) / n_samples
        return grad


class NLLoss(Loss):
    def __init__(self):
        super().__init__()

    @check_shape
    def __call__(self, prds, labs):
        n_samples = prds.shape[0]

        a = n.exp(prds - n.max(prds, axis=1, keepdims=True))
        a = a / n.sum(a, axis=1, keepdims=True)
        c = labs * n.log(a)
        c = - n.sum(c, axis=1)
        c = n.sum(c)
        y = c / n_samples

        return y

    @check_shape
    def grad(self, prds, labs):
        n_samples = prds.shape[0]
        grad = (n.copy(prds) - labs)
        grad = grad / n_samples
        return grad
















