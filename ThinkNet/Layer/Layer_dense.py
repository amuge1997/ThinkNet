from .Layer import AbleTrain
import numpy as n


class Dense(AbleTrain):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.params_init()
        self.grads_init()

    def params_init(self):
        in_features = self.in_features
        out_features = self.out_features
        self.params = {
            'w': n.random.randn(in_features, out_features) / n.sqrt(in_features * out_features),
            'b': n.random.randn(1, out_features) / n.sqrt(1 * out_features),
        }

    def grads_init(self):
        in_features = self.in_features
        out_features = self.out_features
        self.grads = {
            'w': n.zeros((in_features, out_features)),
            'b': n.zeros((1, out_features))
        }

    def forward(self, inps):
        self.inps = inps
        x = inps
        w = self.params['w']
        b = self.params['b']
        y = x @ w + n.tile(b, (x.shape[0], 1))
        self.outs = y
        return y

    def backward(self, grads):
        x = self.inps
        self.grads['w'] = x.T @ grads
        self.grads['b'] = n.sum(grads, axis=0)
        w = self.params['w']
        gz = grads @ w.T
        return gz












