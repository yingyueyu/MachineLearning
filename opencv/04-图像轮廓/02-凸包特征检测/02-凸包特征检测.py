import cv2
import numpy as np

# 创建一组点
points = np.array([[10, 10], [10, 100], [100, 10], [100, 100], [50, 50]])

# 计算凸包
hull = cv2.convexHull(points)

# 绘制凸包
image = np.zeros((200, 200, 3), dtype=np.uint8)
for x, y in points:
    cv2.circle(image, (x, y), 2, (255, 255, 255), thickness=2)
cv2.polylines(image, [hull], True, (0, 255, 0), 2)

# 显示图像
cv2.imshow("Convex Hull", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
