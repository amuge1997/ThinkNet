from .Layer import Layer
import numpy as n

class WeightLayer(Layer):
    def __init__(self):
        super().__init__()
        self.type = 'weight'

class Dense(WeightLayer):
    def __init__(self,in_features,out_features):
        super().__init__()

        self.params = {
            'w': n.random.randn(in_features,out_features)/20,
            'b': n.random.randn(1,out_features)/20,
        }
        self.grads = {}

    def forward(self,inputX):
        self.inputX = inputX

        X = inputX
        W = self.params['w']
        B = self.params['b']
        Y = X @ W + n.tile(B,(X.shape[0],1))
        self.outputY = Y

        return Y

    def backward(self,gradX):
        X = self.inputX
        self.grads['w'] = X.T @ gradX
        self.grads['b'] = n.sum(gradX,axis=0)

        W = self.params['w']
        gradY = gradX @ W.T

        return gradY












