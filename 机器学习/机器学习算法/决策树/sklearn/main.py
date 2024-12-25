import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def random_samples(num):
    samples = np.empty((4, num))
    # - 温度 23~40 之间
    samples[0] = np.random.randint(23, 41, num)
    # - 湿度 50~90 之间
    samples[1] = np.random.randint(50, 91, num)
    # - 风力 1~3 之间
    samples[2] = np.random.randint(1, 4, num)
    # - 光线强度 1~3 之间
    samples[3] = np.random.randint(1, 4, num)
    # return samples.T
    # numpy 中的 transpose 相当于 pytorch 中的 permute
    # return samples.transpose(1, 0), np.random.randint(0, 3, num)
    return samples.transpose(1, 0), np.random.rand(num)


samples, labels = random_samples(20)

# 相关api
# DecisionTreeClassifier 分类器
# DecisionTreeRegressor 回归器


# api 文档：https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#decisiontreeclassifier

# 参数
# criterion: {'gini', 'entropy', 'log_loss'}, default='gini'
#    划分标准，gini代表基尼系数，entropy代表信息增益，log_loss代表对数损失
# splitter: {'best', 'random'}, default='best'，决策树节点分裂策略
#     'random'，意味着决策树节点会随机选择一个特征进行分裂。这种策略在某些情况下可能会提高模型的泛化能力，因为它可以避免过拟合。
# max_depth: 最大深度
# min_samples_split: 最小分裂样本数，默认为 2，小于 min_samples_split，那么这个节点将不会被分裂
# min_samples_leaf: 最小叶子节点样本数，默认为 1，小于等于 min_samples_leaf，那么这个节点将不会被分裂
# min_weight_fraction_leaf: 叶子节点最小权重分数，默认为 0
#     我们可以给所有样本指定权重值，那么在叶节点处，我们可以计算这个节点内样本的权重和，然后除以所有样本的权重和，得到一个比例
#     当该比例小于 min_weight_fraction_leaf 时，那么这个节点将不会被分裂
# max_features: 最大特征数，默认为 None，表示使用所有特征
#     设置整数时: 考虑整数个特征
#     设置浮点数时: 考虑特征总数的百分比
#     'sqrt': 表示所有特征数的平方根，例如一共4个特征，那么就考虑2个特征
#     'log2': 表示所有特征数的log2值，例如一共8个特征，那么就考虑3个特征
# random_state: 随机种子
# max_leaf_nodes: 最大叶子节点数，默认为 None，表示不限制叶子节点数
# min_impurity_decrease: 最小不纯度减少，默认为 0，表示不限制
#     样本携带的信息量，是可以量化的，我们称为信息熵
#     通过算法可以计算一个节点分裂后和分裂前，携带信息量的变化，也就是信息熵的减少量
#     如果分裂后，信息熵减少的值大于 min_impurity_decrease，那么这个节点就会被分裂
# class_weight: 类别权重，默认为 None，表示所有类别的权重相等
# model = DecisionTreeClassifier(
#     criterion='entropy',
#     splitter='best',
#     max_depth=20,
#     min_samples_split=2,
#     min_samples_leaf=2,
#     max_features=4,
#     class_weight={0: 3, 1: 2, 2: 1}
# )

model = DecisionTreeRegressor()

model.fit(samples, labels)

print(labels)
print(model.predict(samples))

print(model.predict(np.array([[35, 88, 2, 3]])))
