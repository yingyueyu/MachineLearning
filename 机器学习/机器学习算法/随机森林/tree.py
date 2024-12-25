# 这是 CART 决策树示例
# CART 决策树以基尼系数 Gini 为标准划分评价标准和维度
# Gini = 1 - (p1^2 + p2^2 + ... + pn^2)
# Gini 描述了在某种概率情况下的不确定性，基尼系数越低，不确定性越低，常以最低基尼系数的划分作为最佳划分标准
# CART（Classification and Regression Trees，分类与回归树）
import numpy as np


# 创建决策树模型的步骤总结
# 1. 创建节点类 DecisionTreeNode
#   1. 保存当前节点的划分标准的特征索引、划分阈值、叶节点特征值（也就是分类结果）
#   2. 保存左右子树
# 2. 创建树类 DecisionTree
#   1. 属性: 包含树的最大深度、树的根节点、
#   2. 方法:
#       1. fit: 通过样本构建树
#       2. _build_tree: 构建树的逻辑
#           1. 判断是否达到了递归的终止条件，终止条件是：样本全部分类、达到最大深度
#           2. 每次递归调用 _find_best_split 寻找最佳划分点，并构建左右子树
#           3. 使用找到的划分点，划分样本
#           4. 递归调用 _build_tree 构建左右子树
#       3. _find_best_split: 寻找最佳划分点，返回分割点的索引和阈值
#           1. 逻辑是遍历每个特征，尝试用当前特征划分样本，并计算当前划分方法下的基尼系数
#           2. 重复这一步骤，直到找到最小的基尼系数对应的特征索引和阈值
#       4. _gini_index: 求节点的基尼系数
#       5. predict: 接收一组新样本，预测其分类
#       6. _predict_tree: 这是 predict 的逻辑代码，递归遍历树，预测新样本的分类


# 1. 创建节点类 DecisionTreeNode
#   1. 保存当前节点的划分标准的特征索引、划分阈值、叶节点特征值（也就是分类结果）
#   2. 保存左右子树
class DecisionTreeNode:
    # feature_idx: 划分标准的特征索引
    # threshold: 划分阈值
    # value: 叶节点特征值
    def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right


# 2. 创建树类 DecisionTree
#   1. 属性: 包含树的最大深度、树的根节点、
#   2. 方法:
#       1. fit: 通过样本构建树
#       2. _build_tree: 构建树的逻辑
#           1. 判断是否达到了递归的终止条件，终止条件是：样本全部分类、达到最大深度
#           2. 每次递归调用 _find_best_split 寻找最佳划分点，并构建左右子树
#           3. 使用找到的划分点，划分样本
#           4. 递归调用 _build_tree 构建左右子树
#       3. _find_best_split: 寻找最佳划分点，返回分割点的索引和阈值
#           1. 逻辑是遍历每个特征，尝试用当前特征划分样本，并计算当前划分方法下的基尼系数
#           2. 重复这一步骤，直到找到最小的基尼系数对应的特征索引和阈值
#       4. _gini_index: 求节点的基尼系数
#       5. predict: 接收一组新样本，预测其分类
#       6. _predict_tree: 这是 predict 的逻辑代码，递归遍历树，预测新样本的分类
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        # 根节点
        self.root = None

    # X: 样本
    # y: 标签
    def fit(self, X, y):
        # 构建树根
        self.root = self._build_tree(X, y, depth=0)

    # 2. _build_tree: 构建树的逻辑
    # 1. 判断是否达到了递归的终止条件，终止条件是：样本全部分类、达到最大深度
    # 2. 每次递归调用 _find_best_split 寻找最佳划分点，并构建左右子树
    # 3. 使用找到的划分点，划分样本
    # 4. 递归调用 _build_tree 构建左右子树
    def _build_tree(self, X, y, depth=0):
        # 1. 判断是否达到了递归的终止条件，终止条件是：样本全部分类、达到最大深度
        if (self.max_depth is not None and depth >= self.max_depth) or len(np.unique(y)) == 1:
            # 此处取样本分类最多的作为本节点的类型（少数服从多数）
            return DecisionTreeNode(value=np.argmax(np.bincount(y)))

        # 2. 每次递归调用 _find_best_split 寻找最佳划分点，并构建左右子树
        feature_idx, threshold = self._find_best_split(X, y)
        # 3. 使用找到的划分点，划分样本
        left_idx = X[:, feature_idx] <= threshold
        right_idx = X[:, feature_idx] > threshold
        left_samples = X[left_idx]
        right_samples = X[right_idx]

        # 若左右分支的一边为空，则将当前节点变成叶节点
        if left_samples.shape[0] == 0 or right_samples.shape[0] == 0:
            return DecisionTreeNode(value=np.argmax(np.bincount(y)))

        left_y = y[left_idx]
        right_y = y[right_idx]

        # 构建左右分支
        # 4. 递归调用 _build_tree 构建左右子树
        left_node = self._build_tree(left_samples, left_y, depth + 1)
        right_node = self._build_tree(right_samples, right_y, depth + 1)
        # 返回当前节点
        return DecisionTreeNode(feature_idx, threshold, left=left_node, right=right_node)

    # 3. _find_best_split: 寻找最佳划分点，返回分割点的索引和阈值
    # 1. 逻辑是遍历每个特征，尝试用当前特征划分样本，并计算当前划分方法下的基尼系数
    # 2. 重复这一步骤，直到找到最小的基尼系数对应的特征索引和阈值
    def _find_best_split(self, X, y):
        # 1. 逻辑是遍历每个特征，尝试用当前特征划分样本，并计算当前划分方法下的基尼系数
        n_feature = X.shape[1]

        # 保存gini最小时的 gini值 特征索引 阈值
        best_gini = float('inf')
        best_feature_idx = None
        best_threshold = None

        # 循环特征
        for feature in range(n_feature):
            # 找出特征下可能的特征值
            feature_values = np.unique(X[:, feature])
            # 循环特征值
            for threshold in feature_values:
                # 根据阈值找出左右分支的样本
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                # 计算基尼系数
                gini = self._gini_index(y[left_idx], y[right_idx])
                # 判断是否获得更好的gini系数
                if gini < best_gini:
                    best_gini = gini
                    best_feature_idx = feature
                    best_threshold = threshold
        # 找到最优解并返回
        return best_feature_idx, best_threshold

    def _gini_index(self, left_y, right_y):
        left_gini = 1 - np.sum((np.bincount(left_y) / len(left_y)) ** 2)
        right_gini = 1 - np.sum((np.bincount(right_y) / len(right_y)) ** 2)
        # 加权平均，求总基尼系数
        # 将 left_gini right_gini 作为系数
        return (left_gini * len(left_y) + right_gini * len(right_y)) / (len(left_y) + len(right_y))

    # 5. predict: 接收一组新样本，预测其分类
    def predict(self, X):
        # 循环每个样本，并从根节点开始分类
        return [self._predict_tree(X[i], self.root) for i in range(X.shape[0])]

    # 给每个样本在指定节点中划分左右并找到分类
    def _predict_tree(self, x, node):
        # 若节点中存在 value，说明此节点是叶节点
        if node.value is not None:
            return node.value
        # 根据特征和阈值判断接下来应该走左右哪个分支
        node = node.left if x[node.feature_idx] <= node.threshold else node.right
        # 递归调用，查看下一个节点中的分支情况
        return self._predict_tree(x, node)
