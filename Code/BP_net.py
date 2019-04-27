import numpy as np
import matplotlib.pyplot as plt


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


def forward(X, W, B):
    """
    :param X: 1*m
    :param W: 权重列表，包含各层之间的权重 w1:1*3 w2:3*1
    :param b:
    :return: z1 a1:3*m  z2 a2:1*m A:[a1,a2]
    """
    A = []
    Z = []
    z0 = W[0].T.dot(X) + B[0]
    Z.append(z0)
    a0 = sigmoid(z0)
    A.append(a0)

    count = 0
    index = 1
    while count < len(W)-1:
        z = W[index].T.dot(A[index-1]) + B[index]
        a = sigmoid(z)
        Z.append(z)
        A.append(a)
        index += 1
        count += 1

    return Z, A


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
        Ei = np.sum(Ei, axis=0)
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


def backward(Layer, X, Y, Z, A, W):
    # 假设 1--3--2
    # delta_List
    delta_list = []

    # 输出层delta : 2*m
    delta_L = -(Y - A[len(A) - 1]) * sigmoid_de(Z[len(Z) - 1])
    delta_list.append(delta_L)

    # #所有单个样本的E对W的梯度的和 : 3*2 和W[len(W)-1]一致
    # E_w_total = A[len(A)-2].dot(delta_L.T)

    # 从第L-1层到第2层依次计算隐藏层的delta,更新参数
    # 4.25 怎么解决角标问题 4.26 用设置多个标记的方法解决
    # 设置各个List的标记
    index_w = len(W) - 1
    index_delta = 0
    index_z = len(Z) - 2

    for i in range(len(Layer) - 2):
        delta_l = (W[index_w].dot(delta_list[index_delta])) * sigmoid_de(Z[index_z])
        delta_list.append(delta_l)
        # 标记更新
        index_w -= 1
        index_delta += 1
        index_z -= 1

    # 存储每一层 所有样本的代价函数E对参数w的偏导数的List
    E_w_total_list = []
    # 代价函数对参数w、b的偏导数
    index_a = len(A) - 1
    for i in range(len(delta_list)):
        if i < len(delta_list) - 1:
            # 所有(N)样本在l层的E对W_l的偏导数的和 : 3*2 和W[len(W)-1]一致
            E_wl_total = A[index_a - 1].dot(delta_list[i].T)
            E_w_total_list.append(E_wl_total)
            index_a -= 1
        # 遍历到了输入层
        elif i == len(delta_list) - 1:
            E_wl_total = X.dot(delta_list[i].T)
            E_w_total_list.append(E_wl_total)

    # #转换为重输入层到输出层
    # E_b_list = list(reversed(delta_list))
    # E_w_total_list = list(reversed(E_w_total_list))
    E_b_list = delta_list
    return E_w_total_list, E_b_list


def update(W, B, E_w_total_list, E_b_list,X,lr):
    N = X.shape[1]
    index = 0
    for i in range(len(W)-1,-1,-1):
        W[i] = W[i] - (lr/N)*E_w_total_list[index]
        # E_b_total = float(np.sum(E_b_list[index],axis=1))
        E_b_total = np.sum(E_b_list[index],axis=1).reshape(E_b_list[index].shape[0],1)
        B[i] = B[i] - (lr/N)*E_b_total

        index += 1

def parameter_init(Layer):
    W = []
    B = []
    for i in range(len(Layer)-1):
        wi = np.random.randn(Layer[i],Layer[i+1])
        bi = np.random.randn(Layer[i+1],1)
        W.append(wi)
        B.append(bi)

    return W,B


if __name__ == '__main__':
    X = np.linspace(0,2*np.pi,150,dtype=float).reshape(1,150)
    #noise = np.random.normal(0, 0.05, X.shape)
    Y = (np.sin(X)+1)/2

    Layer = [1,80,100,80,1]
    W, B = parameter_init(Layer)
    lr = 0.1
    step = 0
    y_loss = []
    while step < 30000:
        Z,A = forward(X,W,B)
        E_w_total_list,E_b_list = backward(Layer,X,Y,Z,A,W)
        update(W,B,E_w_total_list,E_b_list,X,lr)
        # if step%100 == 0:
        #     print("cost:" + str(cost(X, Y, A)))
        x_loss =np.arange(0,step+1,1)

        y_loss.append(cost(X,Y,A))

        if step % 1000 == 0:
            plt.figure(1)
            plt.subplot(1,2,1)
            plt.plot(X, Y,'g*')
            plt.plot(X, A[len(A)-1],'r+')

            plt.subplot(1,2,2)
            plt.plot(x_loss,y_loss,'b+')

            plt.show()

        step += 1


