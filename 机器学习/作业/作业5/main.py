# - 创建一个随机函数 random_samples，用于随机 N 个样本，每个样本有 M 个特征，每个特征是 min ~ max 的值
# - 调用 random_samples 创建 100 个随机分布在长宽高都为 1 的空间中的样本
# - 创建 Kmeans 模型
# - 调用 Kmeans 模型，将样本分为 5 类
# - 将分类结果图绘制出来
import random

import numpy as np
from matplotlib import pyplot as plt


def random_samples(N, M, min, max):
    return min + np.random.rand(N, M) * (max - min)


samples = random_samples(100, 3, -0.5, 0.5)
print(samples)


# kmeans 算法步骤
# 1. 随机 k 个样本作为参考中心点（质心）
# 2. 计算每个样本到参考的距离
# 3. 将样本划分到距离最近的参考上
# 4. 更新参考中心点为每个簇的平均值
# 5. 重复 2-4 直到参考不再变化
class KmeansModel:
    def __init__(self, max_iter=None):
        self.max_iter = max_iter

    def fit(self, X, k):
        # 1. 随机 k 个样本作为参考中心点（质心）
        idx = random.sample(list(np.arange(len(X))), k)
        # 保存质心
        self.centers = X[idx]
        # 构造分类簇
        self.clusters = [[] for i in range(k)]

        self._predict(X)
        return self.clusters, self.centers

    def _distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))
        # return np.linalg.norm(x - y)

    def _predict(self, X, iter_count=0):
        # 达到了最大迭代次数
        if self.max_iter is not None and iter_count >= self.max_iter:
            return

        # 清空簇
        for cluster in self.clusters:
            cluster.clear()

        # 2. 计算每个样本到参考的距离
        for sample in X:
            # 最短距离
            min_dis = float('inf')
            # 最短距离的质心索引
            min_center_idx = None

            # 循环质心
            for i, center in enumerate(self.centers):
                # 计算距离
                dis = self._distance(sample, center)
                if dis < min_dis:
                    min_dis = dis
                    min_center_idx = i

            # 3. 将样本划分到距离最近的参考上
            self.clusters[min_center_idx].append(sample)

        # 缓存质心
        self.centers_cache = self.centers.copy()

        # 4. 更新参考中心点为每个簇的平均值
        for i, cluster in enumerate(self.clusters):
            # 计算簇的平均点
            center = np.array(cluster).mean(axis=0)
            # 更新质心
            self.centers[i] = center

        # 对比更新前后的质心差距，差距不大则跳出递归
        if np.allclose(self.centers, self.centers_cache, rtol=1e-3):
            return

        # 5. 重复 2-4 直到参考不再变化
        self._predict(X, iter_count + 1)


model = KmeansModel()
k = 5
clusters, centers = model.fit(samples, k)

print(clusters)
print(centers)

# 画框
fig = plt.figure()
# 画布
ax = fig.add_subplot(projection='3d')


def random_colors(k):
    r = np.random.randint(128, 255, k)
    g = np.random.randint(128, 255, k)
    b = np.random.randint(128, 255, k)
    colors = np.stack((r, g, b), axis=0)
    colors = [f'#{colors[:, i][0]:02x}{colors[:, i][1]:02x}{colors[:, i][2]:02x}' for i in range(k)]
    return colors


colors = random_colors(k)
print(colors)

# 循环绘制 k 个簇
for i, cluster in enumerate(clusters):
    cls = np.array(cluster)
    ax.scatter(cls[:, 0], cls[:, 1], cls[:, 2], s=64, c=colors[i])

# 绘制质心
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=100, c=colors, edgecolor='#000000')

plt.show()
