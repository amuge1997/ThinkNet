import numpy as n

def data1():
    n_samples = 5
    # print(n.random.randn(n_samples,2) + 1)
    inputX = n.concatenate((n.random.randn(n_samples,2) + 2,n.random.randn(n_samples,2) - 2),axis=0)

    l1 = n.zeros((n_samples,2))
    l1[:,0] = 1.0

    l2 = n.zeros((n_samples,2))
    l2[:,1] = 1.0

    predY = n.concatenate((l1,l2),axis=0)

    return n_samples,inputX,predY



def data2():
    n_samples = 7
    h = 0.5
    inputX = n.array([
        [0, 0],
        [0.5, 0.5],
        [1, 1],
        [1.5, 0.5],
        [2, 0],
        [1, 0],
        [1, 0.5],

        [0, 0 + h],
        [0.5, 0.5 + h],
        [1, 1 + h],
        [1.5, 0.5 + h],
        [2, 0 + h],
        [1, 2.5],
        [1, 3]
    ])

    predY = n.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],
        [1, 0],

        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1]
    ])

    return n_samples,inputX,predY


def data3():
    n_sample = 30
    n_samples = n_sample * 2

    offset = 1.5
    p11 = n.random.normal(0, 0.5, (n_sample, 2)) + offset

    p12 = n.random.normal(0, 0.5, (n_sample, 2)) - offset

    p21 = n.random.normal(0, 0.5, (n_sample, 2)) + offset
    p21[:, 1] -= 2 * offset

    p22 = n.random.normal(0, 0.5, (n_sample, 2)) - offset
    p22[:, 1] += 2 * offset

    inputX = n.concatenate((p11, p12, p21, p22), axis=0)

    l1 = n.zeros((n_samples, 2))
    l1[:, 0] = 1.0

    l2 = n.zeros((n_samples, 2))
    l2[:, 1] = 1.0

    predY = n.concatenate((l1, l2), axis=0)
    return n_samples,inputX,predY

n_samples, inputX, realY = data3()















