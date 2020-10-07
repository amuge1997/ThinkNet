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
            'b': n.random.randn(1, out_features) / n.sqrt(1 * out_features),
        }

    def gradsInit(self):
        in_features = self.in_features
        out_features = self.out_features
        self.grads = {
            'w': n.zeros((in_features,out_features)),
            'b': n.zeros((1,out_features))
        }

    def forward(self, inps):
        self.inputX = inps

        x = inps
        w = self.params['w']
        b = self.params['b']
        y = x @ w + n.tile(b, (x.shape[0], 1))
        self.outputY = y

        return y

    def backward(self, grads):
        X = self.inputX
        self.grads['w'] = X.T @ grads
        self.grads['b'] = n.sum(grads, axis=0)

        w = self.params['w']
        gz = grads @ w.T

        return gz












