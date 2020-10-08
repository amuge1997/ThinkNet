

class Layer:
    def __init__(self):
        self.params = None
        self.grads = None
        self.inps = None
        self.outs = None
        self.is_train = None

    def grads_init(self):
        pass

    def params_init(self):
        pass

    def forward(self, inps):
        raise Exception('Layer is None')

    def backward(self, grads):
        raise Exception('Layer is None')

    def get_params(self):
        return self.params

    def get_grads(self):
        return self.grads


class UnTrain(Layer):
    def __init__(self):
        super().__init__()
        self.is_train = False


class AbleTrain(Layer):
    def __init__(self):
        super().__init__()
        self.is_train = True





