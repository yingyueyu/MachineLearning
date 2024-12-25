# 天气是否适合打篮球的数据
import numpy as np

# 特征代表:
# 天气:
#   0. 晴
#   1. 阴
#   2. 雨
# 温度:
#   0. 低
#   1. 中
#   2. 高
# 湿度:
#   0. 正常
#   1. 高
# 风速:
#   0. 弱
#   1. 强

# 标签代表:
# 0. 不适合打篮球
# 1. 适合打篮球
X = np.array([
    [0, 2, 1, 0],  # 晴, 高, 高, 弱
    [0, 2, 1, 1],  # 晴, 高, 高, 强
    [1, 2, 1, 0],  # 阴, 高, 高, 弱
    [2, 1, 1, 0],  # 雨, 中, 高, 弱
    [2, 0, 0, 0],  # 雨, 低, 正常, 弱
    [2, 0, 0, 1],  # 雨, 低, 正常, 强
    [1, 0, 0, 1],  # 阴, 低, 正常, 强
    [0, 1, 1, 0],  # 晴, 中, 高, 弱
    [0, 0, 0, 0],  # 晴, 低, 正常, 弱
    [2, 1, 0, 0],  # 雨, 中, 正常, 弱
])

y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])


# 朴素贝叶斯模型
class NaiveBayes:
    # 初始化函数中的参数，用到什么就加什么
    def __init__(self):
        pass

    def fit(self, X, y):
        # - 计算每个类的先验概率 \( P(c_i) \)。
        self.prior = np.bincount(y) / y.shape[0]
        # - 计算每个特征在给定类条件下的条件概率 \( P(x_j|c_i) \)。

        # 特征个数
        self.n_feature = X.shape[1]

        # 所有分类的值
        self.all_c = np.unique(y)
        # 分类数量
        self.n_c = self.all_c.shape[0]

        # 构造一个用于存储条件概率的容器
        # key: 分类的值，value: 当前分类下的特征条件概率
        # {
        #     0: [
        #         [0.5, 0.3, 0.2], # 天气的概率分布
        #         [0.8, 0.2] # 湿度的概率分布
        #     ],
        #     1: []
        # }
        self.feature_prob = {c: [] for c in self.all_c}

        # 循环每种分类值，计算每种分类下每种特征条件的概率
        for c in self.all_c:
            # 取出符合分类的样本
            sample = X[y == c]
            # 循环所有特征
            for feature in range(self.n_feature):
                # 取出符合分类的样本中，该特征的所有值
                _sample = sample[:, feature]
                # 计算当前特征有几个特征值
                feature_value = np.unique(_sample)
                # 计算当前分类下的条件概率
                # 后面 +1 实现拉普拉斯平滑
                tmp = np.bincount(_sample, minlength=len(feature_value)) + 1
                prob = tmp / tmp.sum()
                self.feature_prob[c].append(prob)

    def predict(self, X):
        # - 对于待分类样本 \( \mathbf{x} \)，计算每个类的后验概率 \(P(c_i |\mathbf{x}) \)。

        result = []

        # 循环所有类别
        for i in range(self.n_c):
            # 构造每个分类的概率结果
            # 使用 1 作为概率初始值
            r = np.ones(X.shape[0])

            # 为了获取特征概率，需要字典中的 key，c 就是 key
            c = self.all_c[i]
            # 获取先验概率
            prior = self.prior[i]

            r *= prior

            for feature in range(self.n_feature):
                # 获取对应特征的值
                samples = X[:, feature]
                # 获取对应分类下对应特征概率分布
                # 并将samples作为索引取出概率值
                feature_prob = self.feature_prob[c][feature][samples]

                r *= feature_prob

            result.append(r)

        result = np.array(result).T

        # - 选择后验概率最大的类作为预测结果。

        # 计算最大概率的分类结果
        c = np.argmax(result, axis=1)

        return result, c


model = NaiveBayes()

model.fit(X, y)

prob, c = model.predict(X)

print(c)
print(y)
