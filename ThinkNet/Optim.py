import numpy as n


class Optimizer:
    def __init__(self, lr):
        self.lr = lr
        self.layers = None
        self.epoch = 0

    def set_net(self, layers):
        self.layers = layers

    def opt_fn(self):

        self.epoch += 1

        layers = self.layers

        # 将所有参数数组扁平化，注意这里所有参数数组均使用拷贝
        # 方式一：可以使用 np.ravel() 进行优化，但由于 np.concatenate 未完全理解其对内存的影响，因此仍然先暂时选择拷贝方式
        # 方式二：不需要 layersParams 只需要在更新参数时使用 += 操作即可
        layers_params = []
        layers_grads = []
        for layer in layers:
            if layer.is_train:
                dc_p = layer.get_params()
                dc_g = layer.get_grads()

                for k in dc_p:
                    layers_params.append(dc_p[k])
                    layers_grads.append(dc_g[k])

        flatten_params = n.concatenate([params.flatten() for params in layers_params])
        flatten_grads = n.concatenate([grads.flatten() for grads in layers_grads])

        # 根据梯度计算参数更新值
        flatten_steps = self._opt(flatten_grads)
        flatten_params += flatten_steps

        # 将更新值赋回至参数
        params_n = 0
        for layer in layers:
            if layer.is_train:
                dc_p = layer.get_params()
                for k in dc_p:
                    kp = k
                    vp = dc_p[k]
                    n_len = vp.reshape(-1).shape[0]
                    dc_p[kp] = flatten_params[params_n:params_n+n_len].reshape(vp.shape)
                    params_n += n_len

    # 被调用于获取优化值
    def _opt(self, grads):
        raise Exception('Optim is None')


class Adam(Optimizer):
    def __init__(self, lr):
        super().__init__(lr)
        self.p1 = 0.9
        self.p2 = 0.999
        self.e = 1e-8

        self.s = 0
        self.r = 0

    def _opt(self, grads):
        self.s = self.p1 * self.s + (1 - self.p1) * grads
        self.r = self.p2 * self.r + (1 - self.p2) * grads ** 2

        s = self.s / (1 - self.p1 ** self.epoch)
        r = self.r / (1 - self.p2 ** self.epoch)

        ret = - self.lr * s / (n.sqrt(r) + self.e)

        return ret


class GD(Optimizer):
    def __init__(self, lr):
        super().__init__(lr)

    def _opt(self, grads):
        ret = - self.lr * grads
        return ret
















