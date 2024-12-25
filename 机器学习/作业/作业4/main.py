# 随机两个队伍的数据
import random

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

# 人数
N = 20

# 坐标
points = np.random.rand(N, 2)

# 标签
# 0蓝队 1红队
labels = np.zeros(N)
idx = random.sample(list(np.arange(N)), int(N * 0.5))
labels[idx] = 1

# 随机一个受测点
idx = np.random.randint(0, N)
print(idx)

points = list(points)
A = points.pop(idx)
labels = list(labels)
A_label = labels.pop(idx)
points = np.array(points)
labels = np.array(labels)


class KNN:
    def __init__(self, A, points, labels):
        self.A = A
        self.points = points
        self.labels = labels

        # 计算每个点到 A 的距离
        self.distances = {self._distance(A, p): labels[i] for i, p in enumerate(points)}
        # 按照距离排序
        self.distances = sorted(self.distances.items(), key=lambda item: item[0])

    def _distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def predict(self, k):
        # 取 k 个值
        samples = self.distances[:k]
        # 获取分类
        tmp = np.array([int(s[1]) for s in samples])
        # 统计分了出现的次数
        tmp = np.bincount(tmp, minlength=2)
        # 计算概率分布
        pro = tmp / np.sum(tmp)
        # 取最大值索引
        c = np.argmax(pro)
        return pro, c


model = KNN(A, points, labels)

k = 0

# 用循环试探，找到最小的 k 值
while True:
    # 一定要修改循环条件 k 否则死循环跳不出去
    k += 1

    # 3. k 若大于等于 N，则没有找到能够以多打少的 k 值，就直接推出
    if k >= N:
        k = -1
        break

    pro, c = model.predict(k)
    # 1. 正好红蓝概率分布都是 0.5
    if pro[0] == pro[1] == 0.5:
        break
    # 2. 预测正确 A 属于哪个队伍的前提下，求出一方概率大于另一方
    if c != A_label:
        continue
    if pro[c] > 0.5 and np.all(pro != 0):
        break

print(k)

fig, ax = plt.subplots()

sc = ax.scatter(points[:, 0], points[:, 1], 64, c=labels, cmap='bwr')
ax.scatter(A[0], A[1], 64, c='g', edgecolor='b' if A_label == 0 else 'r', linewidth=3)

# 只有当找到 k 值，才绘制圈
if k != -1:
    # 找出对应 k 值最远的点，并把距离作为半径花圈
    dis = model.distances[:k][-1]
    r = dis[0]
    # 创建圆圈补丁
    circle = patches.Circle(A, r, edgecolor='orange', fill=False, linewidth=1)
    # 打补丁
    ax.add_patch(circle)

plt.colorbar(sc)
plt.axis('equal')
plt.show()
