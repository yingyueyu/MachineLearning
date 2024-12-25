import numpy as np
from matplotlib import pyplot as plt


# 随机样本点
# num: 个数
# c: 类型 0 为红点 1 为蓝点
def rand_point(num, c):
    # 随机生成 3 维的 num 个点
    # 前两个代表 xy 坐标， 最后一个代表分类
    points = np.random.rand(num, 3)
    # 将类别改为 c
    points[:, -1] = c
    return points


# 随机一个用于检测的样本点
x = np.random.rand(2)
red_points = rand_point(10, 0)
blue_points = rand_point(10, 1)
points = np.concatenate((red_points, blue_points))


# 声明模型
class KNN:
    # k: 选取的最近邻的个数
    def __init__(self, k=3):
        self.k = k

    # 拟合 训练: 学习数据中的特征和规律
    def fit(self, x, data):
        # 分离点的坐标和类别
        points = data[:, :-1]
        labels = data[:, -1]

        # 1）计算测试数据与各个训练数据之间的距离；
        # 构造字典，key: 距离，value: 标签
        distances = {self.distance(x, p): labels[idx] for idx, p in enumerate(points)}

        # 2）按照距离的递增关系进行排序；
        distances = sorted(distances.items(), key=lambda item: item[0])

        # 3）选取距离最小的K个点；
        tmp = distances[:self.k]

        # 4）确定前K个点所在类别的出现频率；
        tmp = np.array([int(c) for dis, c in tmp])
        # 统计每种类别出现的次数
        counter = np.bincount(tmp)

        # 5）返回前K个点中出现频率最高的类别作为测试数据的预测分类。
        # 计算概率分布
        pro = counter / np.sum(counter)
        # 取概率最大的类别
        idx = np.argmax(pro)

        return pro, idx

    # 欧式距离
    def distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2, axis=0))


model = KNN(3)
pro, c = model.fit(x, points)
print(pro)
print(c)

# 画图
fig, ax = plt.subplots()

sc = ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='RdBu')
ax.scatter(x[0], x[1], c='g')

plt.colorbar(sc)
plt.show()
