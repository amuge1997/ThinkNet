from .Layer import UnTrain
import numpy as n


class Relu(UnTrain):
    def __init__(self):
        super().__init__()

    def forward(self, inps):
        self.inps = inps
        x = inps
        y = n.maximum(x, 0.0)
        self.outs = y
        return y

    def backward(self, grads):
        x = self.inps
        gz = grads * n.where(x > 0.0, 1.0, 0.0)
        return gz


class Sigmoid(UnTrain):
    def __init__(self):
        super().__init__()

    @staticmethod
    def sigmoid(inps):
        y = 1.0 / (1.0 + n.exp(- inps))
        return y

    def forward(self, inps):
        self.inps = inps
        x = inps
        y = self.sigmoid(x)
        self.outs = y
        return y

    def backward(self, grads):
        x = self.inps
        gz = grads * self.sigmoid(x) * (1 - self.sigmoid(x))
        return gz













