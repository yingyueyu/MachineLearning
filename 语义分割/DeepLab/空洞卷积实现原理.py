import torch
import matplotlib.pyplot as plt

input_conv = torch.ones((3, 3))


def conv2d_t(input_conv, kernel, stride=1, padding=0):
    m = kernel.shape[0]  # 5
    n = input_conv.shape[0]  # 3
    l = (n - 1) * stride + m - 2 * padding  # 7
    conv_result = torch.ones((l, l))
    for i in range(n):
        for j in range(n):
            conv_result[i:i + m, j:j + m] += input_conv[i, j] * kernel
    return conv_result


def dilate_kernel(kernel, dilation):
    kernel_size = kernel + (kernel - 1) * (dilation - 1)
    conv_kernel = torch.zeros((kernel_size, kernel_size))
    conv_kernel[0::dilation, 0::dilation] = 1
    return conv_kernel


def dilate_conv_3(a, b, c):
    kernel1 = dilate_kernel(3, a)
    kernel2 = dilate_kernel(3, b)
    kernel3 = dilate_kernel(3, c)
    result = conv2d_t(input_conv, kernel1)
    result = conv2d_t(result, kernel2)
    result = conv2d_t(result, kernel3)
    return result


ax1 = plt.subplot(121)
result = dilate_conv_3(2, 4, 8)
ax1.imshow(result)
ax2 = plt.subplot(122)
result = dilate_conv_3(1, 2, 9)
ax2.imshow(result)
plt.show()
