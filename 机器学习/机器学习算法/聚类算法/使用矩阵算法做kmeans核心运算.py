import numpy as np

samples = np.arange(10)

print(np.random.choice(samples, 5, replace=False))

# points = np.array([
#     [[1, 2], [1, 2]],
#     [[3, 4], [3, 4]],
#     [[5, 6], [5, 6]],
# ])

points = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
])

# 扩展维度
points = np.expand_dims(points, axis=1)
print(points.shape)
# 复制维度
# points = np.tile(points, (1, 2, 1))
points = np.repeat(points, 2, axis=1)
print(points.shape)
print(points)

# centers = np.array([
#     [[0, 1], [1, 2]],
#     [[0, 1], [1, 2]],
#     [[0, 1], [1, 2]],
# ])

centers = np.array([
    [0, 1], [1, 2]
])

centers = np.expand_dims(centers, axis=0)
print(centers.shape)
centers = np.repeat(centers, 3, axis=0)
print(centers.shape)
print(centers)

# 求距离
dis = np.linalg.norm(points - centers, axis=2)
print(dis)

# tmp = (points - centers) ** 2
# tmp = np.sum(tmp, axis=2)
# tmp = np.sqrt(tmp)
# print(tmp)

# 求每个样本对质心距离的最小值索引
idx = np.argmin(dis, axis=1)
print(idx)
