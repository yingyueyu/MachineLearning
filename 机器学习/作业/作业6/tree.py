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
import numpy as np


class DecisionTreeNode:
    def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None, depth=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right
        self.depth = depth


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth=0):
        # 1. 判断是否达到了递归的终止条件，终止条件是：样本全部分类、达到最大深度
        if (self.max_depth is not None and depth >= self.max_depth) or len(np.unique(y)) == 1:
            return DecisionTreeNode(value=np.argmax(np.bincount(y)))
        # 2. 每次递归调用 _find_best_split 寻找最佳划分点，并构建左右子树
        feature_idx, threshold = self._find_best_split(X, y)
        # 3. 使用找到的划分点，划分样本
        left_idx = X[:, feature_idx] <= threshold
        right_idx = X[:, feature_idx] > threshold
        left_samples = X[left_idx]
        right_samples = X[right_idx]
        left_y = y[left_idx]
        right_y = y[right_idx]
        # 4. 递归调用 _build_tree 构建左右子树
        left_node = self._build_tree(left_samples, left_y, depth=depth + 1) if len(left_y) > 0 else None
        right_node = self._build_tree(right_samples, right_y, depth=depth + 1) if len(right_y) > 0 else None
        return DecisionTreeNode(feature_idx=feature_idx, threshold=threshold, left=left_node, right=right_node,
                                depth=depth)

    def _find_best_split(self, X, y):
        n_feature = X.shape[1]

        best_gini = float('inf')
        best_feature_idx = None
        best_threshold = None

        for feature in range(n_feature):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold
                gini = self._gini(y[left_idx], y[right_idx])
                if gini < best_gini:
                    best_gini = gini
                    best_feature_idx = feature
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def _gini(self, left_y, right_y):
        left_gini = 1 - np.sum((np.bincount(left_y) / len(left_y)) ** 2)
        right_gini = 1 - np.sum((np.bincount(right_y) / len(right_y)) ** 2)
        return (left_gini * len(left_y) + right_gini * len(right_y)) / (len(left_y) + len(right_y))

    def predict(self, X):
        return [self._predict_tree(X[i], self.root) for i in range(X.shape[0])]

    def _predict_tree(self, x, node):
        if node.value is not None:
            return node.value
        next_node = node.left if x[node.feature_idx] <= node.threshold else node.right
        return self._predict_tree(x, next_node)
