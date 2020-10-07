from Read_Mnist import load_labels,load_images
import numpy as np
from ThinkNet.Core import Sigmoid, Dense,Net,NLLoss,GD,MSELoss, toOnehot,Adam, Relu
from sklearn.metrics import accuracy_score
from ThinkNet.Layer_conv import Conv2d, Flatten

IntType = np.int64


def one_hot_encoding(labels, num_class=None):
    """one_hot_encoding Create One-Hot Encoding for labels

    Arguments:
        labels {np.ndarray or list} -- The original labels

    Keyword Arguments:
        num_class {int} -- Number of classses. If None, automatically
            compute the number of calsses in the given labels (default: {None})

    Returns:
        np.ndarray -- One-hot encoded version of labels
    """

    if num_class is None:
        num_class = np.max(labels) + 1
    one_hot_labels = np.zeros((len(labels), num_class))
    one_hot_labels[np.arange(len(labels)), labels] = 1
    return one_hot_labels.astype(IntType)


def load_data():
    import cv2 as cv
    x_ls = []
    for i in load_images('MNIST/t10k-images.idx3-ubyte'):
        x = cv.resize(i, (24, 24))
        x = x[np.newaxis, ...]
        x_ls.append(x)
    data_x = np.concatenate(x_ls)
    data_y = load_labels('MNIST/t10k-labels.idx1-ubyte')
    return data_x, data_y


if __name__ == '__main__':

    print('loading data #####')
    train_X, train_y_ = load_data()
    n_samples = train_X.shape[0]
    train_X = (train_X - 127) / 128
    train_X = train_X[:, np.newaxis, ...]
    train_y = one_hot_encoding(train_y_.astype('int'))

    print('loading data complete #####')
    print()

    is_train = True
    layers = [
        Conv2d(filters=8, kernel_size=3, input_shape=[1, 24, 24]),
        Relu(),
        Conv2d(filters=8, kernel_size=2, stride=(2, 2), is_padding=False, input_shape=[8, 24, 24]),
        Relu(),

        Conv2d(filters=16, kernel_size=3, input_shape=[8, 12, 12]),
        Relu(),
        Conv2d(filters=16, kernel_size=2, stride=(2, 2), is_padding=False, input_shape=[16, 12, 12]),
        Relu(),

        Conv2d(filters=32, kernel_size=2, stride=(2, 2), is_padding=False, input_shape=[16, 6, 6]),
        Relu(),
        Conv2d(filters=32, kernel_size=1, stride=(1, 1), input_shape=[32, 3, 3]),
        Relu(),
        Conv2d(filters=1, kernel_size=1, stride=(1, 1), input_shape=[32, 3, 3]),
        Relu(),

        Flatten(),

        Dense(1 * 3 * 3, 10),
        Sigmoid(),

        # Conv2d(filters=2, kernel_size=3, stride=(3, 3), is_padding=False, input_shape=[1, 16, 16]),
        # Relu(),
        #
        # Conv2d(filters=2, kernel_size=3, stride=(3, 3), is_padding=False, input_shape=[2, 9, 9]),
        # Relu(),
        #
        # Conv2d(filters=2, kernel_size=1, stride=(1, 1), is_padding=False, input_shape=[2, 3, 3]),
        # Relu(),
        #
        # Conv2d(filters=2, kernel_size=1, stride=(1, 1), is_padding=False, input_shape=[2, 3, 3]),
        # Relu(),
        #
        # Flatten(),
        #
        # Dense(2 * 3 * 3, 10),
        # Sigmoid(),

    ]

    mse = MSELoss()
    nll = NLLoss()
    gd = GD(1e-3)
    adam = Adam(1e-3)

    net = Net(layers, nll, adam)

    epochs = 500
    batch_size = 32
    for epoch in range(epochs):
        randi = np.random.randint(0, train_X.shape[0], (batch_size, ))

        batch_x = train_X[randi]
        batch_y = train_y[randi]

        predL = net.forward(batch_x, batch_y)

        ls = []
        predd = toOnehot(predL)
        for i in predd:
            ls.append(np.argmax(i))
        batch_yy = train_y_[randi]
        acc = accuracy_score(batch_yy, ls)

        if (epoch+1) % 1 == 0:
            loss = np.sqrt(np.sum((predL - batch_y) ** 2)) / batch_x.shape[0]
            print('epoch {}:{}'.format(epoch+1,loss))
            print('epoch {}:{}'.format(epoch+1,acc))
            print()
        net.backward()
        net.update()

    n_test = 256
    inputX = train_X[0:n_test]

    pred = net.predict(inputX)
    pred = toOnehot(pred)
    ls = []
    for i in pred:
        ls.append(np.argmax(i))

    realY = train_y_[0:n_test]
    acc = accuracy_score(realY, ls)
    print(ls)
    print(realY)
    print(acc)

    # 要想进行对比，必须尽量减少全连接单元数量，如只要10到20个，否则即使卷积层随机参数并禁止训练，网络仍然有出色的性能

    # 卷积 不训练 e200 b32 lr3 0.0546875,0.09765625,0.1484375
    # 卷积 训练   e200 b32 lr3 0.3984375,0.35546875,0.3828125

    # 卷积 不训练 e500 b32 lr3 0.078125,0.25,0.23828125
    # 卷积 训练   e500 b32 lr3 0.66015625,0.65625

    # f4,k5+f8,k3+f16,k1 e1000 b32 lr3 loss0.984375




