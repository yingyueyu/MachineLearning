# api包括：
# KNeighborsClassifier
# KNeighborsRegressor
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# 加载鸢尾花数据集
ds = load_iris()
# data: 数据本身
# target: 数据标签
data = ds['data']
target = ds['target']

print(data)
print(data.shape)
print(target)
print(target.shape)

# 划分数据集，返回训练和测试用的数据
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=0)

print(X_train)
print(Y_train)


# 创建 KNN 模型

# 创建 KNN 分类器
# api 文档: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#kneighborsclassifier
# n_neighbors: 邻居数量，K近邻的K值
# weights: 邻居权重，'uniform'表示所有邻居的权重相同，'distance'表示邻居的权重与距离成反比
#    也能填一个函数，接受一个距离数组，并返回一个包含权重的相同形状的数组。
# algorithm: 近邻搜索方法，'auto' 'ball_tree'、'kd_tree'、'brute'
# leaf_size: 如果使用'ball_tree'或'kd_tree'，则表示叶子节点的大小
# p: 距离度量(其实是 metric 公式中的参数)，1表示曼哈顿距离，2表示欧氏距离
#    曼哈顿距离公式：d(x, y) = \sum_{i=1}^{n} |x_i - y_i|；这是求两者特征值之差的绝对值之和。
# metric: 距离度量，'minkowski'表示闵可夫斯基距离，p=1时为曼哈顿距离，p=2时为欧氏距离
#    minkowski距离公式：d(x, y) = (\sum_{i=1}^{n} |x_i - y_i|^p)^(1/p)
#    若不填默认值，则需要指定一个计算距离的函数
# metric_params: 传给自定义距离度量函数的参数
# n_jobs: 并行计算的数量，-1表示使用所有可用的处理器

# 自定义权重函数
def weights(distances):
    # 返回权重值
    # return distances
    # 返回 uniform
    # return np.ones_like(distances)
    # 返回 distance
    dis = np.array(distances)
    dis = 1 - dis
    return dis


def my_metric(x, y, a, b, c):
    # x 代表第一个样本
    # y 代表第二个样本
    # a、b、c 是 metric_params 中的参数
    dis = np.sqrt(np.sum((x - y) ** 2))
    # 返回距离
    return dis


# 分类模型
# model = KNeighborsClassifier(
#     n_neighbors=5,
#     # weights='uniform',
#     weights=weights,
#     # algorithm='ball_tree',
#     algorithm='auto',
#     leaf_size=1,
#     p=2,
#     # metric='minkowski',
#     metric=my_metric,
#     metric_params={'a': 66, 'b': 'hello', 'c': False},
#     n_jobs=4
# )

# 回归模型
model = KNeighborsRegressor()

# 训练模型
model.fit(X_train, Y_train)

# 预测
print(model.predict(X_test))
print(Y_test)
