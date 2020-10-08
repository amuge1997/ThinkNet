

class Net:
    def __init__(self, layers, loss, opt):
        self.layers = layers
        self.loss = loss
        self.opt = opt
        self.inps = None
        self.labs = None

        self.opt.set_net(layers)

    def predict(self, inps):
        layers = self.layers
        x = inps
        for layer in layers:
            y = layer.forward(x)
            x = y

        result = x
        return result

    def forward(self, inps, labs):
        self.inps = inps
        self.labs = labs

        layers = self.layers
        x = inps
        for layer in layers:
            x = layer.forward(x)
        result = x
        return result

    def backward(self):
        labs = self.labs
        loss = self.loss
        layers = self.layers

        prds = layers[-1].outs
        grad = loss.grad(prds, labs)
        for layer in layers[::-1]:
            grad = layer.backward(grad)

    def update(self):
        opt = self.opt
        opt.opt_fn()













