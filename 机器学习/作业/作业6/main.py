import numpy as np

from tree import DecisionTree

np.random.seed(100)


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
    return samples.transpose(1, 0), np.random.randint(0, 3, num)


samples, labels = random_samples(500)
print(samples)
print(samples.shape)

trees = [DecisionTree(10) for i in range(5)]

# 随机取值的方法:
# 方法一: 循环取值
# all_idx = list(np.arange(500))
#
# def rand_one(all_idx):
#     idx = np.random.randint(0, len(all_idx))
#     one = all_idx.pop(idx)
#     return one, all_idx
#
# # 循环取值
# i, all_idx = rand_one(all_idx)


# 方法二: 用掩码矩阵取值
idx_metrix = np.ones(500) * -1
print(idx_metrix)


# 打印节点内容
def print_node(node):
    # 叶节点打印值
    if node.value is not None:
        print(f'value: {node.value}')
        return

    # 打印当前节点的阈值和特征
    print(f'feature: {node.feature_idx}; threshold: {node.threshold}; depth: {node.depth}')

    # 打印左右子树
    if node.left is not None:
        print_node(node.left)
    if node.right is not None:
        print_node(node.right)


for i in range(len(trees)):
    # 查找可以使用的坑位
    mask = idx_metrix == -1
    # np.nonzero 求 mask 中非零值的索引
    nonzero_idx = np.nonzero(mask)[0]
    # 随机 90 个可用的索引
    idx = np.random.choice(nonzero_idx, 90, replace=False)
    # 赋值占位，下一次就不能使用这个索引了
    idx_metrix[idx] = i
    # 获取索引位置上的训练样本和训练标签
    train_samples = samples[idx]
    train_labels = labels[idx]
    trees[i].fit(train_samples, train_labels)
    print(f'tree: {i}')
    print_node(trees[i].root)

# 获取测试样本
test_samples = samples[idx_metrix == -1]
test_labels = labels[idx_metrix == -1]
print(test_samples)
print(test_samples.shape)

result = []

# 测试
for tree in trees:
    result.append(tree.predict(test_samples))

# 转置矩阵，让每行结果的含义变成: 5 个决策树对一个样本的预测结果
result = np.array(result).T
print(result)

pro_result = []

# 遍历每个样本的结果
for i in range(result.shape[0]):
    tmp = np.bincount(result[i], minlength=3)
    pro = tmp / 5
    pro_result.append(pro)

pro_result = np.array(pro_result)
print(pro_result)
