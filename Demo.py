from ThinkNet import Sigmoid, Relu, Dense, Net, NLLoss, GD, MSELoss, to_onehot, to_softmax, Adam
from Data_Samples import inputX, realY, n_samples
import numpy as n
import matplotlib.pyplot as p
import time
import matplotlib as mpl

if __name__ == '__main__':

    # 网络结构
    layers = [
        Dense(2, 10),
        Relu(),
        Dense(10, 10),
        Relu(),
        Dense(10, 2),
        Sigmoid()
    ]
    # 损失函数
    nll = NLLoss()
    mse = MSELoss()
    # 优化器
    adam = Adam(1e-3)
    gd = GD(1e-3)

    # 网络模型构建
    net = Net(layers, mse, adam)

    dc = {
        'fd': 0.0,
        'bd': 0.0,
        'up': 0.0
    }

    # 训练
    epochs = 5000
    for epoch in range(epochs):
        start = time.time()
        predY = net.forward(inputX, realY)
        end = time.time()
        runTime = end - start
        dc['fd'] += runTime
        # print('前向运行时间:', runTime)

        if (epoch+1) % 500 == 0:
            print('epoch {} loss {}'.format((epoch+1),nll(predY, realY)))
        start = time.time()
        net.backward()
        end = time.time()
        runTime = end - start
        dc['bd'] += runTime
        # print('后向运行时间:', runTime)

        start = time.time()
        net.update()
        end = time.time()
        runTime = end - start
        dc['up'] += runTime
        # print('更新运行时间:', runTime)

    sumTime = dc['fd'] + dc['bd'] + dc['up']
    fdrate = dc['fd'] / sumTime
    bdrate = dc['bd'] / sumTime
    uprate = dc['up'] / sumTime

    print('前向运行时间: {}s {}%'.format(round(dc['fd'], 2), round(fdrate * 100, 2)))
    print('反向运行时间: {}s {}%'.format(round(dc['bd'], 2), round(bdrate * 100, 2)))
    print('更新运行时间: {}s {}%'.format(round(dc['up'], 2), round(uprate * 100, 2)))
    print('总时间: {}'.format(round(sumTime,2)))

    # 预测
    predY = net.predict(inputX)
    # print(predY.shape)
    # print(to_onehot(predY))

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













