from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

# 除了使用 SVC 外，还有以下api
# LinearSVC: 线性支持向量分类器，等价于 SVC kernel='linear' 时，但内部部分默认参数和 SVC 不同
# SVR: 用于回归问题的支持向量机
# LinearSVR: 用于回归问题的线性支持向量机，等价于 SVR kernel='linear' 时，但内部部分默认参数和 SVR 不同
# NuSVC: 用于多分类问题的支持向量机，与 SVC 类似，但使用 nu 而不是 C 作为正则化参数
#    的 nu 参数是一个在 (0) 和 (1) 之间的浮点数，表示支持向量的比例。具体来说，nu 参数控制了模型中支持向量的数量
#    例如，如果 nu=0.1，那么模型中大约有10%的样本是支持向量。
# NuSVR: 用于回归问题的支持向量机，与 SVR 类似，但使用 nu 而不是 C 作为正则化参数
# OneClassSVM: 用于异常检测的支持向量机，的基本思想是找到一个超球面（对于线性核）或超椭球面（对于非线性核），使得大部分正常数据点位于这个超球面或超椭球面内部，而异常值位于外部。
#    异常值: 与正常数据点对比，规律更小的数据点，常被当做异常或噪声


X, y = make_blobs(n_samples=10, centers=2, random_state=0)
y = [1 if i == 0 else -1 for i in y]

# api 文档: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# 参数
# C: 正则化参数，C值越大，对误分类的惩罚越大，模型越复杂，容易过拟合
# kernel: 核函数，默认为rbf，可选参数有linear、poly、rbf、sigmoid、precomputed
# degree: 多项式核函数的阶数
# gamma: 核函数系数(高斯核分母参数)，默认为scale，可选参数有auto、scale、float
#   scale: 则它使用 1 / （n_features * X.var（）） 作为 gamma 的值 （X.var() 是X的方差）
#   auto: 则使用 1 / n_features
#   float: 任意非负数
# coef0: 核函数中的常数项。它仅在“poly”和“sigmoid”中显着。
# shrink: 在支持向量机（SVC）中，shrinking 参数用于控制是否使用收缩启发式算法。收缩启发式算法可以加速训练过程并提高模型的泛化能力。
# 具体来说，当 shrinking=True 时，SVC 在训练过程中会使用一种称为“收缩”的技术。这种方法通过在每次迭代中只更新一部分支持向量来减少计算量，从而提高训练速度。
# 当 shrinking=False 时，SVC 将不使用收缩启发式算法，这意味着每次迭代都会更新所有样本。
# 默认情况下，shrinking 参数被设置为 True，因此大多数情况下你不需要手动设置它。
# probability: 是否使用概率估计，默认为False。是否启用概率估计。这必须在调用拟合之前启用，这将减慢该方法的速度，因为它在内部使用 5 倍交叉验证，并且predict_proba可能与预测不一致
# tol: 停止训练的容忍度。如果迭代次数超过max_iter，或者损失函数的下降小于tol，则训练停止。
# cache_size: 指定内核缓存大小（以MB为单位）。内核缓存用于存储计算出的核矩阵，以便在后续的迭代中重用，从而加快训练速度。
# class_weight: 类别权重。如果指定，则按指定的权重分配类别。如果未指定，则所有类别的权重都相等。
# verbose: 是否启用详细输出。如果设置为 True，则模型将输出训练过程中的详细信息，如迭代次数、损失函数值等。
# max_iter: 最大迭代次数。如果达到最大迭代次数，则训练停止。
# decision_function_shape: 决策函数形状(涉及多分类问题)。可选参数有ovo、ovr、None。默认为ovr。如果设置为ovo，则模型将使用一对一策略计算决策函数。如果设置为ovr，则模型将使用一对多策略计算决策函数。如果设置为None，则模型将使用原始的决策函数。
# break_ties: （决定决策函数返回内容，计算成本较高）若为true，则decision_function_shape='ovr'，且类数> 2，则返回类别的得分，而不是预测的类别。如果类数= 2，则返回单个得分。如果为false，则decision_function_shape='ovr'，且类数> 2，则返回预测的类别。如果为false，则decision_function_shape='ovr'，且类数= 2，则返回预测的类别。
# random_state: 随机种子。用于设置随机数生成器的种子，以便在多次运行时获得可重复的结果。
model = SVC(
    C=3.,
    kernel='sigmoid',
    # kernel='rbf',
    # kernel='poly',
    # kernel='linear',
    degree=10,
    coef0=1,  # 多项式核中的常数项
    # gamma
    shrinking=True,
    probability=True,
    cache_size=100,
    class_weight={-1: 1, 1: 1},
    verbose=True,
    max_iter=1000,
    break_ties=False,
    random_state=0
)

model.fit(X, y)

# 决策函数
print(model.decision_function(X))
# 计算支持向量，接近 1 的就是支持向量
print(y * model.decision_function(X))
# support_vectors_ 属性 查看支持向量 支持向量越少越好
print(model.support_vectors_)
print(X)
# b 值
print(model.intercept_)
# w 值
# print(model.coef_)
print(model.coef0)

# 预测
# print(model.predict(X))
# print(y)


fig, ax = plt.subplots()
sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
# 画支持向量
sv = model.support_vectors_
ax.scatter(sv[:, 0], sv[:, 1], c='#000', marker='x')

# 显示决策边界教程：https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-py
# 显示决策边界
DecisionBoundaryDisplay.from_estimator(
    model,
    X,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
    ax=ax,
)

plt.colorbar(sc)
plt.show()
