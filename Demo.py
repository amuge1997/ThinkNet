
from ThinkNet.Core import Sigmoid,Relu,Dense,Net,NLLoss,GD,MSELoss,toSoftmax,toOnehot,Adam
from Data_Samples import inputX,realL,n_samples
import numpy as n,matplotlib.pyplot as p


layers = [
    Dense(2,64),
    Relu(),
    Dense(64,64),
    Relu(),
    Dense(64,2),

]

mse = MSELoss()
nll = NLLoss()
gd = GD(1e-3)
adam = Adam(1e-3)

net = Net(layers, nll, gd)

epochs = 5000
for i in range(epochs):

    predL = net.forward(inputX, realL)
    if i%100 == 0:
        print(nll(predL, realL))
    net.backward()
    net.update()

predL = net.predict(inputX)
# print(predL)
# print(toSoftmax(predL))

x = n.arange(-5,5,0.05)
y = n.arange(-5,5,0.05)
xx,yy = n.meshgrid(x,y)
megX = n.concatenate((xx.reshape(-1,1),yy.reshape(-1,1)),axis=1)

predY = net.predict(megX)


ls0 = []
ls1 = []
ls = []
for i in range(megX.shape[0]):
    if n.argmax(predY[i]) == 0:
        ls0.append(megX[i])
        ls.append(0)
    elif n.argmax(predY[i]) == 1:
        ls1.append(megX[i])
        ls.append(1)
Z = n.array(ls)

# 方式一
# arr0 = n.array(ls0)
# arr1 = n.array(ls1)
# p.scatter(arr0[:,0],arr0[:,1],c='green')
# p.scatter(arr1[:,0],arr1[:,1],c='yellow')

# 方式二
import matplotlib as mpl
cm_light = mpl.colors.ListedColormap(['green','yellow'])
p.pcolormesh(xx,yy,Z.reshape(xx.shape),cmap=cm_light)

print(toOnehot(predL))

p.scatter(inputX[0:n_samples,0],inputX[0:n_samples,1],c='red')
p.scatter(inputX[n_samples:,0],inputX[n_samples:,1],c='blue')
p.show()






