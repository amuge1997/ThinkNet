import numpy as n


def checkShape(infunc):
    def outfunc(ins,X1,X2):
        if X1.shape != X2.shape:
            raise Exception('数组形式不一致')
        else:
            ret = infunc(ins,X1,X2)
            return ret
    return outfunc


def toSoftmax(arr):
    # arr : n_samples,n_dims

    # 减去 max(arr) 是为了数值稳定但大小相对正确
    a = n.exp(arr - n.max(arr, axis=1, keepdims=True))
    Y = a / n.sum(a, axis=1, keepdims=True)
    return Y


def toOnehot(arr):
    # arr : n_samples,n_dims

    res = n.zeros_like(arr)
    for i in range(arr.shape[0]):
        res[i,n.argmax(arr[i])] = 1.0
    return res


def one_hot_encoding(labels, num_class=None):
    if num_class is None:
        num_class = n.max(labels) + 1
    one_hot_labels = n.zeros((len(labels), num_class))
    one_hot_labels[n.arange(len(labels)), labels] = 1
    return one_hot_labels.astype('int')










