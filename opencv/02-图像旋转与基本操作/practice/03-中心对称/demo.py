import cv2
import numpy as np

src = cv2.imread("assets/example.png")
h, w, c = src.shape
dist = np.zeros(shape=(h * 2, w * 2, c), dtype=np.uint8)
part1 = src
part2 = cv2.flip(part1, 0)  # x轴
part3 = cv2.flip(part1, 1)  # y轴
part4 = cv2.flip(part1, -1)  # xy轴
# 组装
dist[0:h, 0:w] = part1
dist[h:2 * h, 0:w] = part2
dist[0:h, w:2 * w] = part3
dist[h:2 * h, w:2 * w] = part4

cv2.imshow("dist",dist)
cv2.waitKey(0)
cv2.destroyAllWindows()
