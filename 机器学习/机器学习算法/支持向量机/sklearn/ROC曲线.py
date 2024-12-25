from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 生成示例数据
X, y = make_classification(n_samples=100, n_features=5, n_classes=2)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建LinearSVC模型
model = LinearSVC(fit_intercept=True, max_iter=10000)

# 训练模型
model.fit(X_train, y_train)

# 获取决策函数的输出
y_score = model.decision_function(X_test)

print(y_score)

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_score)

print(tpr)
print(fpr)

# 计算AUC
roc_auc = auc(fpr, tpr)

print(roc_auc)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
