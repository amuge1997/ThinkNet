from .Layer import UnTrain
import numpy as n


class Flatten(UnTrain):
    def __init__(self):
        super().__init__()
        self.ori_shape = None

    def forward(self, x):
        self.inps = x
        self.ori_shape = x.shape
        outs = n.reshape(x, [x.shape[0], -1])
        self.outs = outs
        return outs

    def backward(self, grad):
        gz = n.reshape(grad, self.ori_shape)
        return gz


class MaxPool2D(UnTrain):
    def __init__(self, stride=(2, 2)):
        super().__init__()
        self.mask = None
        self.stride = stride

    def forward(self, inputs):
        # inputs : N, C, H, W
        n_samples, channels, height, width = inputs.shape
        stride_h, stride_w = self.stride
        kernel_size_h = stride_h
        kernel_size_w = stride_w
        if ((height - kernel_size_h + 1) - 1) % stride_h != 0 or ((width - kernel_size_w + 1) - 1) % stride_w:
            raise Exception('error')
        sum_h = int(((height - kernel_size_h + 1) - 1) / stride_h + 1)
        sum_w = int(((width - kernel_size_w + 1) - 1) / stride_w + 1)
        outs = n.zeros((n_samples, channels, sum_h, sum_w))
        mask = n.zeros_like(inputs)
        for hi in range(sum_h):
            for wi in range(sum_w):
                anchor_h = hi * stride_h
                anchor_w = wi * stride_w
                patch = inputs[:, :, anchor_h:anchor_h+kernel_size_h, anchor_w:anchor_w+kernel_size_w]
                patch = n.reshape(patch, [n_samples, channels, -1])
                argmax = n.argmax(patch, axis=2)
                col = mask[:, :, anchor_h:anchor_h + stride_h, anchor_w:anchor_w + stride_w]
                col = n.reshape(col, [n_samples, channels, stride_h * stride_w])

                for ni in range(n_samples):
                    for ci in range(channels):
                        col[ni, ci, argmax[ni, ci]] = 1
                        outs[ni, ci, hi, wi] = patch[ni, ci, argmax[ni, ci]]
                col = n.reshape(col, [n_samples, channels, stride_h, stride_w])
                mask[:, :, anchor_h:anchor_h + stride_h, anchor_w:anchor_w + stride_w] = col
        self.mask = mask
        self.outs = outs
        return outs

    def backward(self, grad):
        # grad : N, C, H, W
        mask = self.mask
        stride_h, stride_w = self.stride
        n_samples, channels, sum_h, sum_w = grad.shape
        gz = n.zeros((n_samples, channels, sum_h*stride_h, sum_w*stride_w))
        for hi in range(0, sum_h):
            for wi in range(0, sum_w):
                anchor_h = hi * stride_h
                anchor_w = wi * stride_w
                for ni in range(n_samples):
                    for ci in range(channels):
                        patch = grad[ni, ci, hi, wi] * n.ones((stride_h, stride_w))
                        patch_mask = mask[ni, ci, anchor_h:anchor_h + stride_h, anchor_w:anchor_w + stride_w]
                        patch = patch * patch_mask
                        gz[ni, ci, anchor_h:anchor_h + stride_h, anchor_w:anchor_w + stride_w] = patch
        return gz














