import numpy as n
from .Layer import AbleTrain


# 已测试,应该没问题
def matmul_backward(grad, features_col, kernels_col):
    n_samples, channels, height, width = grad.shape
    grad = n.reshape(grad, [n_samples, channels, height * width])
    grad = n.transpose(grad, [0, 2, 1])
    grad_feature_col = grad @ kernels_col
    grad_kernel_col = features_col.transpose([0, 2, 1]) @ grad
    grad_kernel_col = n.sum(grad_kernel_col, axis=0)
    grad_kernel_col = n.transpose(grad_kernel_col, [1, 0])
    grad_bias = n.sum(n.sum(grad, axis=0), axis=0)
    return grad_feature_col, grad_kernel_col, grad_bias


# 已测试
def matmul_forward(features_col, kernels_col, bias, out_hw):
    out_height, out_width = out_hw
    n_samples = features_col.shape[0]
    ret = features_col @ n.transpose(kernels_col, [1, 0]) + bias
    ret = n.transpose(ret, [0, 2, 1])
    out_channels = ret.shape[1]
    ret = n.reshape(ret, [n_samples, out_channels, out_height, out_width])
    return ret


# 未测试
def grad_col_to_kel(grad_kernels_col, in_channels, kernel_size):
    output_channels = grad_kernels_col.shape[0]
    grad_kernels = n.reshape(grad_kernels_col, [output_channels, in_channels, kernel_size, kernel_size])
    return grad_kernels


# 已测试
def kel_to_col(kernels):
    kernels = n.float32(kernels)
    out_channels, in_channels, kernel_size1, kernel_size2 = kernels.shape
    if kernel_size1 != kernel_size2:
        raise Exception('错误')
    kernel_size = kernel_size1
    ret = n.reshape(kernels, [out_channels, in_channels * kernel_size * kernel_size])
    return ret


# 应该没问题
def grad_col_to_img(features_col, kernel_size, in_channels, in_shape, out_shape, stride):
    n_samples = features_col.shape[0]
    ih = features_col.shape[1]
    height, width = in_shape
    sum_h, sum_w = out_shape
    stride_h, stride_w = stride
    ret = n.zeros((n_samples, in_channels, height, width))
    for ihi in range(ih):
        patch = features_col[:, ihi, :]
        patch = n.reshape(patch, [n_samples, in_channels, kernel_size, kernel_size])
        hi = int(ihi / sum_w)
        wi = ihi % sum_w
        anchor_h = hi * stride_h
        anchor_w = wi * stride_w
        ret[:, :, anchor_h:anchor_h+kernel_size, anchor_w:anchor_w+kernel_size] = patch
    return ret


# 将图像转为向量
def img_to_col(features, kernel_size, stride):
    stride_height, stride_width = stride
    n_samples, channels, in_height, in_width = features.shape
    height_able = in_height - kernel_size + 1
    width_able = in_width - kernel_size + 1
    if (height_able - 1) % stride_height != 0 or (width_able - 1) % stride_width != 0:
        raise Exception('error')
    sum_h = int((height_able - 1) / stride_height) + 1
    sum_w = int((width_able - 1) / stride_width) + 1
    ret = n.zeros((n_samples, sum_h * sum_w, kernel_size**2 * channels))
    for hi in range(sum_h):
        for wi in range(sum_w):
            h_anchor = hi * stride_height
            w_anchor = wi * stride_width
            patch = features[:, :, h_anchor:h_anchor+kernel_size, w_anchor:w_anchor+kernel_size]
            patch = n.reshape(patch, [n_samples, 1, -1])
            ihi = hi * sum_w + wi
            ret[:, ihi:ihi+1, :] = patch
    out_height = sum_h
    out_width = sum_w
    out_hw = (out_height, out_width)
    return ret, out_hw


# 可靠
def re_padding2d(image, pad):
    pad_h, pad_w = pad
    h0 = pad_h[0]
    h1 = -pad_h[1]
    w0 = pad_w[0]
    w1 = -pad_w[1]
    # 以下操作主要是为了防止-0的情况
    if h1 != 0:
        hs = slice(h0, h1)
    else:
        hs = slice(h0, None)
    if w1 != 0:
        ws = slice(w0, w1)
    else:
        ws = slice(w0, None)
    return image[:, :, hs, ws]


# 可靠
def padding2d(image, kernel_size, stride_hw):
    stride_h, stride_w = stride_hw
    n_samples, channels, output_height, output_width = image.shape
    input_height = output_height * stride_h - stride_h + kernel_size
    input_width = output_width * stride_w - stride_w + kernel_size
    pad_sum = input_height - output_height
    pad_half1 = int(pad_sum / 2)
    pad_half2 = pad_sum - pad_half1
    pad_h = (pad_half1, pad_half2)
    pad_sum = input_width - output_width
    pad_half1 = int(pad_sum / 2)
    pad_half2 = pad_sum - pad_half1
    pad_w = (pad_half1, pad_half2)
    ret = n.pad(image, pad_width=[(0, 0), (0, 0), pad_h, pad_w])
    return ret, pad_h, pad_w


class Conv2d(AbleTrain):
    def __init__(self, filters, kernel_size, input_shape, stride=(1, 1), is_padding=True, is_train=True):
        super().__init__()

        self.is_train = is_train

        in_channels, in_height, in_width = input_shape
        self.input_shape = input_shape  # 输入形式
        self.in_channels = in_channels  # 输入通道

        self.out_channels = filters     # 输出通道

        self.ori_in_hw = None           # 原始输入的高宽

        self.kernel_size = kernel_size  # 核尺寸
        self.stride_hw = stride         # 步长

        self.is_padding = is_padding
        self.pad_inputs = None          # 补零后的输入
        self.pad_in_hw = None           # 输入补零后的高宽
        self.pad_out_hw = None          # 根据补零后的输入得到的输出的高宽
        self.pad = None                 # 补零参数,记录了高宽两个维度的补零

        self.params_init()
        self.grads_init()

    def params_init(self):
        out_channels = self.out_channels
        in_channels = self.in_channels
        kernel_size = self.kernel_size
        self.params = {
            'w': n.random.randn(out_channels, in_channels, kernel_size, kernel_size) *
                 (1 / n.sqrt(out_channels + in_channels)),
            'b': n.random.randn(out_channels,) * (1 / n.sqrt(out_channels + in_channels)),
        }

    def grads_init(self):
        out_channels = self.out_channels
        in_channels = self.in_channels
        kernel_size = self.kernel_size
        self.grads = {
            'w': n.zeros((out_channels, in_channels, kernel_size, kernel_size)),
            'b': n.zeros((out_channels,))
        }

    def forward(self, inps):
        self.inps = inps
        self.ori_in_hw = inps.shape[2:]

        kernel_size = self.kernel_size
        kernels = self.params['w']
        bias = self.params['b']
        kernels_col = kel_to_col(kernels)
        stride = self.stride_hw
        # 补零
        if self.is_padding:
            inps, pad_h, pad_w = padding2d(inps, kernel_size, stride)
            self.pad = (pad_h, pad_w)
        else:
            self.pad = ((0, 0), (0, 0))

        # 补零后记录
        self.pad_inputs = inps
        self.pad_in_hw = (inps.shape[2], inps.shape[3])

        features_col, pad_out_hw = img_to_col(inps, kernel_size, stride)      # 图像转向量
        self.pad_out_hw = pad_out_hw
        y = matmul_forward(features_col, kernels_col, bias, self.pad_out_hw)    # 卷积
        self.outs = y
        return y

    def backward(self, grad):
        kernel = self.params['w']
        in_features = self.pad_inputs
        kernel_size = self.kernel_size
        in_channels = self.in_channels
        pad_in_hw = self.pad_in_hw
        pad_out_hw = self.pad_out_hw
        stride = self.stride_hw

        features_col, _ = img_to_col(features=in_features, kernel_size=kernel_size, stride=stride)          # 图像转向量
        kernel_col = kel_to_col(kernel)
        gard_features_col, grad_kernels_col, grad_bias = matmul_backward(grad, features_col, kernel_col)    # 反向传播
        gard_features = grad_col_to_img(        # 梯度向量转梯度图
            gard_features_col,
            kernel_size,
            in_channels,
            pad_in_hw,
            pad_out_hw,
            stride
        )
        grad_kernels = grad_col_to_kel(grad_kernels_col, in_channels, kernel_size)
        if self.is_train:
            self.grads['w'] = grad_kernels
            self.grads['b'] = grad_bias
        gard_features = re_padding2d(gard_features, self.pad)
        return gard_features


if __name__ == '__main__':
    # main3()

    # from SB_MNIST import load_MNIST
    # data_x, data_y = load_MNIST()

    # print(data_x.shape)
    #
    # one = n.float32(data_x[0][n.newaxis, n.newaxis, ...])
    #
    # conv = Conv2d(3, 1, 2)
    # x = conv.forward(one)
    # print(x.shape, x.dtype)

    # inps = n.array(
    #     [
    #         [
    #             [
    #                 [5, 2, 1, 2],
    #                 [3, 4, 3, 4],
    #             ],
    #             [
    #                 [1, 2, 1, 2],
    #                 [3, 4, 3, 4],
    #             ],
    #         ],
    #         [
    #             [
    #                 [1, 2, 9, 2],
    #                 [3, 4, 3, 4],
    #             ],
    #             [
    #                 [1, 2, 1, 2],
    #                 [3, 4, 3, 4],
    #             ],
    #         ],
    #     ]
    # )
    # print(inps.shape)
    #
    # mp = MaxPool2D()
    # mp.forward(inps)
    #
    # grads = mp.backward(n.ones((2, 2, 1, 2)))
    # print(grads)

    pass


















