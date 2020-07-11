import numpy as n
import copy

class Optimizer:
    def __init__(self,lr):
        self.lr = lr
        self.layers = None
        self.epoch = 0


    def setNet(self,layers):
        self.layers = layers

    def optim(self):

        self.epoch += 1

        layers = self.layers

        # 将所有参数数组扁平化，注意这里所有参数数组均使用拷贝
        # 方式一：可以使用 np.ravel() 进行优化，但由于 np.concatenate 未完全理解其对内存的影响，因此仍然先暂时选择拷贝方式
        # 方式二：不需要 layersParams 只需要在更新参数时使用 += 操作即可
        layersParams = []
        layersGrads = []
        for layer in layers:
            if layer.type is 'weight':
                dcP = layer.getParams()
                dcG = layer.getGrads()

                for k in dcP:
                    layersParams.append(dcP[k])
                    layersGrads.append(dcG[k])

        flattenParams = n.concatenate([params.flatten() for params in layersParams])
        flattenGrads = n.concatenate([grads.flatten() for grads in layersGrads])

        # 根据梯度计算参数更新值
        flattenSteps = self._optim(flattenGrads)
        flattenParams += flattenSteps

        # 将更新值赋回至参数
        paramsn = 0
        for layer in layers:
            if layer.type is 'weight':
                dcP = layer.getParams()
                for k in dcP:
                    kp = k
                    vp = dcP[k]
                    nlen = vp.reshape(-1).shape[0]
                    dcP[kp] = flattenParams[paramsn:paramsn+nlen].reshape(vp.shape)
                    paramsn += nlen

    # 被调用于获取优化值
    def _optim(self,grads):
        raise Exception('Optim is None')


class Adam(Optimizer):
    def __init__(self,lr):
        super().__init__(lr)
        self.p1 = 0.9
        self.p2 = 0.999
        self.e = 1e-8

        self.s = 0
        self.r = 0


    def _optim(self,grads):
        self.s = self.p1 * self.s + (1 - self.p1) * grads
        self.r = self.p2 * self.r + (1 - self.p2) * grads ** 2

        s = self.s / (1 - self.p1 ** self.epoch)
        r = self.r / (1 - self.p2 ** self.epoch)

        ret = - self.lr * s / (n.sqrt(r) + self.e)

        return ret


class GD(Optimizer):
    def __init__(self,lr):
        super().__init__(lr)

    def _optim(self,grads):
        # print('optim_')
        # print(grads)
        # print(self.lr)
        ret = - self.lr * grads
        # print(ret)
        return ret
















