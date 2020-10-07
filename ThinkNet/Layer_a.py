from .Layer import Layer
import numpy as n


class ActivationLayer(Layer):
    def __init__(self):
        super().__init__()
        self.type = 'activation'


class Relu(ActivationLayer):
    def __init__(self):
        super().__init__()

    def forward(self, inps):
        self.inputX = inps
        x = inps
        y = n.maximum(x, 0.0)
        self.outputY = y
        return y

    def backward(self, grads):
        x = self.inputX
        gz = grads * n.where(x > 0.0, 1.0, 0.0)
        return gz


class Sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__()

    def sigmoid(self, inps):
        y = 1.0 / (1.0 + n.exp(- inps))
        return y

    def forward(self, inps):
        self.inputX = inps
        x = inps
        y = self.sigmoid(x)

        self.outputY = y

        return y

    def backward(self, grads):
        x = self.inputX
        gz = grads * self.sigmoid(x) * (1 - self.sigmoid(x))
        return gz













