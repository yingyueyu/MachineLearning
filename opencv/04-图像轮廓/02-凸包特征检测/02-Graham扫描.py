import numpy as np
import matplotlib.pyplot as plt
import math

x = np.arange(0, 10).reshape(10, 1).astype(np.int32)
y = np.random.normal(10, 5, size=(10, 1)).astype(np.int32)
points = np.concatenate((x, y), axis=1)

plt.subplot(121)
h, w = points.shape
plt.plot(points[:, 0], points[:, 1], 'ro')

# Graham扫描：1) 排序 2) 扫描
# 排序
points = points.tolist()
# Y值最小为起点
points = sorted(points, key=lambda x: x[1])
p_0 = points[0]
point_theta = []
for point in points:
    if p_0 == point:
        continue
    plt.plot([p_0[0], point[0]], [p_0[1], point[1]], 'b--')
    p_x = point[0] - p_0[0]
    p_y = point[1] - p_0[1]
    angle = math.asin(p_y / (p_x ** 2 + p_y ** 2) ** 0.5) * 180 / math.pi
    if p_x < 0:
        angle += 90
    point_theta.append([*point, int(angle)])
point_theta = sorted(point_theta, key=lambda x: (x[2], (x[0] ** 2 + x[1] ** 2)))
convex_hull = [p_0, point_theta[0][:2]]
index = 0
for p_x, p_y, _ in point_theta[1:]:
    p_0 = convex_hull[index]
    p_1 = convex_hull[index + 1]
    l1_x, l1_y = p_1[0] - p_0[0], p_1[1] - p_0[1]
    l2_x, l2_y = p_x - p_0[0], p_y - p_0[1]
    cross_product = l2_x * l1_y - l1_x * l2_y
    print(cross_product)
    if cross_product < 0:
        convex_hull.append([p_x, p_y])
        index += 1
    else:
        convex_hull.pop(-1)
        convex_hull.append([p_x, p_y])

convex_hull.append(point_theta[-1][:2])
convex_hull = np.array(convex_hull)
plt.subplot(122)
plt.plot(convex_hull[:, 0], convex_hull[:, 1], 'go')
plt.show()
