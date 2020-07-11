from ThinkNet.Core import Sigmoid, Relu, Dense, Net, NLLoss, GD, MSELoss, toSoftmax, toOnehot, Adam
from Data_Samples import inputX, predY, n_samples
import numpy as n, matplotlib.pyplot as p
import matplotlib as mpl

if __name__ == '__main__':

    # 网络结构
    layers = [
        Dense(2, 64),
        Relu(),
        Dense(64, 64),
        Relu(),
        Dense(64, 2),
    ]
    # 损失函数
    nll = NLLoss()
    # 优化器
    adam = Adam(1e-3)

    # 网络模型构建
    net = Net(layers, nll, adam)

    # 训练
    epochs = 1000
    for epoch in range(epochs):
        predY = net.forward(inputX, predY)
        if epoch % 100 == 0:
            print(nll(predY, predY))
        net.backward()
        net.update()

    # 预测
    predY = net.predict(inputX)
    print(predY)

    # 网格区域绘制
    x = n.arange(-5, 5, 0.05)
    y = n.arange(-5, 5, 0.05)
    xx, yy = n.meshgrid(x, y)
    megX = n.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)

    predY = net.predict(megX)

    zls = []
    for row in range(megX.shape[0]):
        if n.argmax(predY[row]) == 0:
            zls.append(0)
        elif n.argmax(predY[row]) == 1:
            zls.append(1)
    Z = n.array(zls)

    cm_light = mpl.colors.ListedColormap(['green', 'yellow'])
    p.pcolormesh(xx, yy, Z.reshape(xx.shape), cmap=cm_light)

    # 数据样本绘制
    p.scatter(inputX[0:n_samples, 0], inputX[0:n_samples, 1], c='red')
    p.scatter(inputX[n_samples:, 0], inputX[n_samples:, 1], c='blue')
    p.show()













