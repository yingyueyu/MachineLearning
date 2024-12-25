import numpy as np
import matplotlib.pyplot as plt


def sample_fu(n, m):
    return np.random.rand(n, m) * n


k = 5
data = sample_fu(100, 3)
keys = data[np.random.choice(range(len(data)), k, replace=False)]  # choice只能用一维数组

for _ in range(10):
    # 分别求每个点到key的距离（用了广播机制，可以打印出来看看，shape=50,5,意思是50个点分别到5个质点的距离）
    distances = np.linalg.norm(data[:, None] - keys, axis=2)
    # 分簇，shape=50，值是0~4
    labels = np.argmin(distances, axis=1)

    # 解释一下难理解的地方，点是用下标区分的，比如第1个点到第2个质点的距离是distances[0,1]，假设到第3个质点的距离最短则labels[0]=2
    # 假设第2个点到第5个质点的距离最短则labels[2]=4

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    color = ['red', 'blue', 'cyan', 'green', 'yellow']
    for i in range(k):
        p = data[i == labels]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], color=color[i])
        ax.scatter(keys[:, 0], keys[:, 1], keys[:, 2], c="black", marker='o', s=100, label='Centroids')
        for j in p:  # :(
            ax.plot([j[0], keys[i, 0]], [j[1], keys[i, 1]],
                    [j[2], keys[i, 2]], 'k--')
    # 以上都是画图

    new_keys = np.array([data[i == labels].mean(axis=0) for i in range(k)])  # 求各簇中心点

    # 判断是否结束（求出来的质点与上一次的一样）
    if np.all(keys == new_keys):
        print('收敛')
        break
    keys = new_keys
    print('不收敛')

plt.show()