from .Layer import Layer
import numpy as n

class NeuronLayer(Layer):
    def __init__(self):
        super().__init__()
        self.type = 'neuron'

class Dense(NeuronLayer):
    def __init__(self,in_features,out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.paramsInit()
        self.gradsInit()

    def paramsInit(self):
        in_features = self.in_features
        out_features = self.out_features
        self.params = {
            'w': n.random.randn(in_features, out_features) / n.sqrt(in_features * out_features),
            'b': n.random.randn(1, out_features) / n.sqrt(in_features * out_features),
        }

    def gradsInit(self):
        in_features = self.in_features
        out_features = self.out_features
        self.grads = {
            'w': n.zeros((in_features,out_features)),
            'b': n.zeros((1,out_features))
        }

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












