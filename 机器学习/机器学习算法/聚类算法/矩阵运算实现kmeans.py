import numpy as np
from matplotlib import pyplot as plt

# kmeans 算法步骤
# 1. 随机 k 个样本作为参考中心点（质心）
# 2. 计算每个样本到参考的距离
# 3. 将样本划分到距离最近的参考上
# 4. 更新参考中心点为每个簇的平均值
# 5. 重复 2-4 直到参考不再变化

X = np.random.rand(100, 2)


class KMeans:
    def __init__(self, k, max_iter=None):
        self.k = k
        self.max_iter = max_iter

    def fit(self, samples):
        # 1. 随机 k 个样本作为参考中心点（质心）
        # 随机不重复的质心索引
        center_idx = np.random.choice(np.arange(len(samples)), self.k, replace=False)
        # 质心
        self.centers = samples[center_idx]
        # 递归
        self._predict(samples)
        # 分化簇
        self.clusters = []
        for i in range(self.k):
            mask = self.labels == i
            self.clusters.append(samples[mask])

    # 5. 重复 2-4 直到参考不再变化
    def _predict(self, samples, iter_count=0):
        if self.max_iter is not None and iter_count >= self.max_iter:
            return

        # 2. 计算每个样本到参考的距离
        # (100, 2)
        # (100, k, 2)
        _samples = np.repeat(np.expand_dims(samples, axis=1), self.k, axis=1)
        # (k, 2)
        # (100, k, 2)
        centers = np.repeat(np.expand_dims(self.centers, axis=0), samples.shape[0], axis=0)
        # 求距离
        distances = np.linalg.norm(_samples - centers, axis=2)
        # 3. 将样本划分到距离最近的参考上
        # 给质心做一个编号，相当于索引值，从 [0, k)
        # 求每一组中最小距离的索引
        self.labels = np.argmin(distances, axis=1)

        self.centers_cache = self.centers.copy()

        # 4. 更新参考中心点为每个簇的平均值
        for i in range(self.k):
            # 求出当前质心索引的布尔张量
            # 作为掩码
            mask = self.labels == i
            # 求平均
            center = np.mean(samples[mask], axis=0)
            # 更新质心
            self.centers[i] = center

        # 对比更新前后的质心是否一致
        if np.allclose(self.centers, self.centers_cache, rtol=1e-3):
            return

        self._predict(samples, iter_count + 1)


def rand_colors(k):
    r = np.random.randint(128, 255, k)
    g = np.random.randint(128, 255, k)
    b = np.random.randint(128, 255, k)
    colors = np.stack((r, g, b), axis=0)
    colors = [f'#{colors[:, i][0]:02x}{colors[:, i][1]:02x}{colors[:, i][2]:02x}' for i in range(k)]
    return colors


k = 5

colors = rand_colors(k)

model = KMeans(k)
model.fit(X)

centers = model.centers
_colors = [colors[i] for i in model.labels]

clusters = model.clusters

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=_colors)
ax.scatter(centers[:, 0], centers[:, 1], c=colors, s=100, edgecolor='#000000')

plt.show()
