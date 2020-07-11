from .Layer import Layer
import numpy as n


class ActivationLayer(Layer):
    def __init__(self):
        super().__init__()
        self.type = 'activation'


class Relu(ActivationLayer):
    def __init__(self):
        super().__init__()

    def forward(self,inputX):
        self.inputX = inputX
        X = inputX
        Y = n.maximum(X,0.0)
        self.outputY = Y
        return Y

    def backward(self,gradX):
        X = self.inputX
        gradY = gradX * n.where(X > 0.0, 1.0, 0.0)
        return gradY


class Sigmoid(ActivationLayer):
    def __init__(self):
        super().__init__()

    def sigmoid(self,inputX):
        Y = 1.0 / (1.0 + n.exp(- inputX))
        return Y

    def forward(self,inputX):
        self.inputX = inputX
        X = inputX
        Y = self.sigmoid(X)

        self.outputY = Y

        return Y

    def backward(self,gradX):
        X = self.inputX
        gradY = gradX * self.sigmoid(X) * ( 1 - self.sigmoid(X) )
        return gradY













