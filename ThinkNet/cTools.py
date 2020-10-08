import numpy as n


def to_softmax(arr):
    # arr : n_samples,n_dims

    # 减去 max(arr) 是为了数值稳定但大小相对正确
    a = n.exp(arr - n.max(arr, axis=1, keepdims=True))
    y = a / n.sum(a, axis=1, keepdims=True)
    return y


def to_onehot(arr):
    # arr : n_samples,n_dims

    res = n.zeros_like(arr)
    for i in range(arr.shape[0]):
        res[i, n.argmax(arr[i])] = 1.0
    return res


def one_hot_encoding(labels, num_class=None):
    if num_class is None:
        num_class = n.max(labels) + 1
    one_hot_labels = n.zeros((len(labels), num_class))
    one_hot_labels[n.arange(len(labels)), labels] = 1
    return one_hot_labels.astype('int')










