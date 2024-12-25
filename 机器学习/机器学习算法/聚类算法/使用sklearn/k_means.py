import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# api 文档: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


X = np.random.rand(100, 2)

# 参数:
# n_clusters: 簇的数量
# init: 初始化方法，可以是'k-means++'或'random'
#   'k-means++': 使用k-means++算法初始化聚类中心，该方法有效加速了收敛速度，步骤如下：
#       1. 随机选择一个数据点作为第一个聚类中心。
#       2. 对于每个数据点，计算它到最近的聚类中心的距离的平方。
#       3. 根据这些距离的概率分布，随机选择下一个聚类中心。
#           假设: 每个点到聚类中心的距离和称为 total 的话，每个点到聚类的距离为 dis
#           概率: dis / total
#           然后随机一个数，看落在哪个概率区间内，就选哪个点作为下一个聚类中心。
#       4. 重复步骤2和3，直到选择了所有的聚类中心。
#   'random': 随机选择数据点作为聚类中心。
# n_init: 初始化聚类中心的次数，默认为10。每次初始化都会计算一个k-means结果，最后取最好的结果。
#    评价指标: 惯性（inertia）或簇内平方和（within-cluster sum of squares）
#    惯性是指每个数据点到其所属簇的质心的距离的平方和。惯性越小，说明数据点在簇内的分布越紧密，聚类效果越好。
# max_iter: 最大迭代次数，默认为300。如果达到最大迭代次数，算法仍未收敛，则停止迭代。
# tol: 收敛阈值，默认为1e-4。如果两次迭代之间的惯性变化小于该阈值，则认为算法已经收敛，停止迭代。
# verbose: 是否打印详细信息，默认为0。如果设置为1，则打印每次迭代的信息。
# random_state: 随机种子，默认为None。
# copy_x: 是否复制输入数据，默认为True。如果设置为False，则直接在输入数据上进行聚类，可能会影响输入数据的原始值。
# algorithm: 聚类算法，默认为'auto'。可选的算法有'auto', 'full', 'elkan'。
#    'auto': 根据数据集的特征自动选择算法。如果数据集的稀疏特征向量稀疏度低于0.1，则选择'elkan'算法，否则选择'full'算法。
#    'full': 使用传统的k-means算法。
#    'elkan': 使用elkan k-means算法，该算法在计算距离时使用了三角不等式，可以加速收敛。
model = KMeans(
    n_clusters=5,
    init='k-means++',
    n_init=5,
    max_iter=10,
    tol=1e-3,
    verbose=True,
    random_state=100,
    copy_x=False,
    algorithm='lloyd'
)

model.fit(X)

# 质心
centers = model.cluster_centers_
print(centers)

# 每个样本属于哪个簇
print(model.labels_)

fig, ax = plt.subplots()

colors = ['r', 'y', 'b', 'g', 'orange']

for i in range(5):
    # 获取 i 对应标签的布尔张量
    mask = model.labels_ == i
    ax.scatter(X[mask][:, 0], X[mask][:, 1], c=colors[i])
ax.scatter(centers[:, 0], centers[:, 1], s=100, c=colors, edgecolor='#000000')

plt.show()
