import numpy as np

lr = 1e-2

# 构造输入
X = np.array([[i + 1, i + 2] for i in range(10)])
y = np.array([[X[i, 0] * X[i, 1], X[i, 0] * X[i, 1]] for i in range(X.shape[0])])
print(X)
print(y)

# 初始化参数
W = np.zeros(2)
b = 0


# 声明模型
def model(X):
    # W (2,)
    _W = W.reshape(1, -1)  # (1, 2)
    # 重复某个维度的值
    _W = np.repeat(_W, X.shape[0], axis=0)  # (10, 2)
    # X (10, 2)
    return _W * X + b


def MSELoss(_y, y):
    return np.sum((_y - y) ** 2)


for i in range(100):
    # 1. 清空梯度
    grad_w = None
    grad_b = None
    # 2. 模型预测
    _y = model(X)
    # 3. 计算损失值
    loss = MSELoss(_y, y)
    print(loss)
    # 4. 计算梯度
    # W 的导数
    dy_to_dloss = 2 * (_y - y)
    dW_to_dy = X
    grad_w = np.mean((dy_to_dloss * dW_to_dy), axis=0)
    # b 的导数
    db_to_dy = 1
    grad_b = np.mean((dy_to_dloss * db_to_dy), axis=0)
    # 5. 更新参数
    W = W - lr * grad_w
    b = b - lr * grad_b.mean()

print('训练结束')
print(model(X))
