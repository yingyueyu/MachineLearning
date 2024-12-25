import numpy as np
import matplotlib.pyplot as plt

# points = np.random.normal(10, 5, size=(10, 2)).astype(np.int32)
x = np.arange(0, 10).reshape(10, 1).astype(np.int32)
y = np.random.normal(10, 5, size=(10, 1)).astype(np.int32)
points = np.concatenate((x, y), axis=1)

plt.subplot(121)
h, w = points.shape
plt.plot(points[:, 0], points[:, 1], 'ro')

# 穷举法原理
convex_hull = []
points = points.tolist()
for i in range(len(points)):
    p1 = points[i]
    is_aside = False
    for j in range(len(points)):
        p2 = points[j]
        if p1 == p2:
            continue
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b--')
        cross_product = []
        for point in points:
            if p1 == point or p2 == point:
                continue
            l1_x = p2[0] - p1[0]
            l1_y = p2[1] - p1[1]
            l2_x = point[0] - p1[0]
            l2_y = point[1] - p1[1]
            # 叉积法
            cross_product.append(l2_x * l1_y - l1_x * l2_y)
        if np.all(np.array(cross_product) > 0):
            convex_hull.append(p2)

convex_hull = np.array(convex_hull)
convex_hull = np.unique(convex_hull, axis=0)

plt.subplot(122)
plt.plot(convex_hull[:, 0], convex_hull[:, 1], 'go')
plt.show()
