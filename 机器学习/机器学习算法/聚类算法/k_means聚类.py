# kmeans 算法步骤
# 1. 随机 k 个样本作为参考中心点（质心）
# 2. 计算每个样本到参考的距离
# 3. 将样本划分到距离最近的参考上
# 4. 更新参考中心点为每个簇的平均值
# 5. 重复 2-4 直到参考不再变化
import random

import numpy as np
from matplotlib import pyplot as plt


# K-means 模型
class KmeansModel:
    def __init__(self, max_iter=None):
        self.max_iter = max_iter
        self.fig, self.ax = plt.subplots()
        # 用于保存画的所有点
        self.scatters = []

    # 训练
    def fit(self, X, k):
        # 1. 随机 k 个样本作为参考中心点（质心）
        # 随机索引
        idx = random.sample(list(np.arange(len(X))), k)
        # 质心
        self.centers = X[idx]
        # 簇 clusters
        # i: 质心的索引
        self.clusters = {i: [] for i in range(k)}

        # 随机颜色值
        self.colors = self.rand_colors(k)

        # 画初始点
        self.scatters.append(self.ax.scatter(X[:, 0], X[:, 1], c='b'))
        # 画质心
        self.scatters.append(
            self.ax.scatter(self.centers[:, 0], self.centers[:, 1], 100, c=self.colors, edgecolor='#000000'))

        # 循环
        self._predict(X)
        print('kmeans 执行结束')

        plt.show()

        # 返回质心和分类好的簇
        return self.centers, self.clusters

    def rand_colors(self, k):
        # 随机rgb三色
        r = np.random.randint(128, 255, k)
        g = np.random.randint(128, 255, k)
        b = np.random.randint(128, 255, k)
        colors = np.stack([r, g, b], axis=0)
        colors = [f'#{colors[:, i][0]:02x}{colors[:, i][1]:02x}{colors[:, i][2]:02x}' for i in range(k)]
        return colors

    # 清空画布
    def clear_canvas(self):
        for scatter in self.scatters:
            scatter.remove()
        self.scatters.clear()

    # 写一个递归函数，将 2~4 步 包装起来
    # iter_count: 循环递归的次数
    def _predict(self, X, iter_count=0):
        # 循环条件检测
        if self.max_iter is not None and iter_count >= self.max_iter:
            return

        # 清空簇，方便重新分配
        for i, cluster in self.clusters.items():
            cluster.clear()

        # 2. 计算每个样本到参考的距离
        for sample in X:
            # 最小距离
            min_dis = float('inf')
            # 最短距离的质心索引
            min_center_idx = None

            # 循环质心
            for i, center in enumerate(self.centers):
                # 求距离
                dis = self._distance(sample, center)
                if dis < min_dis:
                    min_dis = dis
                    min_center_idx = i
            # 将样本归类
            # 3. 将样本划分到距离最近的参考上
            self.clusters[min_center_idx].append(sample)

        # 缓存更新前的质心
        self.centers_cache = self.centers.copy()

        # 更新质心
        # 4. 更新参考中心点为每个簇的平均值
        for i, cluster in self.clusters.items():
            center = np.array(cluster).mean(axis=0)
            self.centers[i] = center

        # 清空画布
        self.clear_canvas()
        # 绘制样本点
        for i, cluster in self.clusters.items():
            np_c = np.array(cluster)
            self.scatters.append(self.ax.scatter(np_c[:, 0], np_c[:, 1], c=self.colors[i]))
        # 绘制质心
        self.scatters.append(
            self.ax.scatter(self.centers[:, 0], self.centers[:, 1], 100, c=self.colors, edgecolor='#000000'))
        plt.pause(0.8)

        # 比较更新前后的质心差距，若足够小，我们认为两者相同，则跳出递归
        if np.allclose(self.centers_cache, self.centers, rtol=1e-3):
            return

        # 递归循环
        # 5. 重复 2-4 直到参考不再变化
        self._predict(X, iter_count + 1)

    def _distance(self, x, y):
        # 求欧式距离
        return np.linalg.norm(x - y)


# 随机 100 个点
X = np.random.rand(100, 2)

model = KmeansModel()

model.fit(X, 5)
