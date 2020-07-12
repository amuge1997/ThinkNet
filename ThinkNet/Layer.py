

class Layer:
    def __init__(self):
        self.params = None
        self.grads = None
        self.inputX = None
        self.outputY = None
        self.type = None

    def gradsInit(self):
        pass

    def paramsInit(self):
        pass

    def forward(self,inputX):
        raise Exception('Layer is None')

    def backward(self,gradX):
        raise Exception('Layer is None')

    def getParams(self):
        return self.params

    def getGrads(self):
        return self.grads








