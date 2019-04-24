import numpy as np

Layer = [1,3,1]
def sigmoid(z):
    """
    激活函数sigmoid
    :param z:
    :return:
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_de(z):
    """
    求sigmoid函数的导数
    :param z:
    :return:
    """
    return sigmoid(z) * (1 - sigmoid(z))


def forward(X, W, b):
    """
    :param X: 1*m
    :param W: 权重列表，包含各层之间的权重 w1:1*3 w2:3*1
    :param b:
    :return: z1 a1:3*m  z2 a2:1*m A:[a1,a2]
    """
    A = []
    Z = []
    for i in range(len(W)):
        z = W[i].T.dot(X) + b
        a = sigmoid(z)
        Z.append(z)
        A.append(a)

    return Z,A


def cost_single(target, net_out):
    """
    对一个训练数据计算代价函数
    :param target:
    :param net_out:
    :return: Ei:1*m
    """
    loss = target - net_out
    Ei = 1 / 2 * (loss * loss)
    # 输出层神经元数量大于1
    if target.shape[0] > 1:
        # axis=0压缩行 axis=1压缩列
        np.sum(Ei, axis=0)
    return Ei


def cost(X, Y, A):
    """
    所有训练数据的总体（平均）代价
    :param X: 1*m
    :param Y: 1*m 目标输出
    :param A: [a1,a2] a2:实际输出 1*m
    :return: E_total: float
    """
    # 样本数量
    N = X.shape[1]
    # A列表的最后一个元素是前向传播的输出结果，即a2
    Ei = cost_single(Y, A[len(A) - 1])
    # 压缩列，因为列数代表样本个数，即把所有单个样本的代价加在一起
    E_total = Ei.sum(axis=1) * 1 / N
    # E_total: List[]
    return float(E_total)

def backward(Layer, Y, Z, A, W, B):
    #假设 1--3--2
    #delta_List
    delta_list = []

    #输出层delta : 2*m
    delta_L = -(Y-A[len(A)-1])*sigmoid_de(Z[len(Z)-1])
    delta_list.append(delta_L)

    #所有单个样本的E对W的梯度的和 : 3*2 和W[len(W)-1]一致
    E_w_total = A[len(A)-2].dot(delta_L.T)

    #从第L-1层到第2层依次计算隐藏层的delta,更新参数
    #4.25 怎么解决角标问题
    for i in range(len(Layer)-2,0,-1):
        delta_l = W[i].dot(delta_list[-i]) * sigmoid_de(Z[-i])
