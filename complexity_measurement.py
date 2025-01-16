from math import pow

def standard_convolution(k, C, N, H, W):
    return pow(k, 2) * C * N * H * W


def depthwise_convolution(k, C, N, H, W):
    return pow(k, 2) * C * H * W + C * N * H * W



if __name__ == '__main__':
    k = 3
    C = 8
    N = 16
    H = 224
    W = 224

    print(standard_convolution(k, C, N, H, W))
    print(depthwise_convolution(k, C, N, H, W))