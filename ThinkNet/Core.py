
import numpy as n
from .cTools import checkShape, toOnehot, toSoftmax
from .Loss import NLLoss, MSELoss
from .Layer_w import Dense
from .Layer_a import Relu, Sigmoid
from .Optim import GD, Adam


class Net:
    def __init__(self, layers, loss, optim):
        self.layers = layers
        self.loss = loss
        self.optim = optim
        self.inputX = None
        self.realL = None

        self.optim.setNet(layers)

    def predict(self, inps):
        layers = self.layers
        x = inps
        for layer in layers:
            y = layer.forward(x)
            x = y

        result = x
        return result

    def forward(self, inps, labs):
        self.inputX = inps
        self.realL = labs

        layers = self.layers
        x = inps
        for layer in layers:
            x = layer.forward(x)
        result = x
        return result

    def backward(self):
        realL = self.realL
        loss = self.loss
        layers = self.layers

        predL = layers[-1].outputY

        grad = loss.grad(predL,realL)
        for layer in layers[::-1]:
            grad = layer.backward(grad)

    def update(self):
        optim = self.optim
        optim.optim()













