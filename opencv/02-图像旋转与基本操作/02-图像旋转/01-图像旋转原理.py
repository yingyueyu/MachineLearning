import numpy as np
import cv2 as cv
import math

# ------------------旋转原理--------------------------------
img = cv.imread("../assets/example.png")
new_img = np.zeros(img.shape, dtype=img.dtype)
print(new_img.shape)
h, w, c = img.shape

# 旋转角度
theta = 30 / 180 * math.pi
# 旋转因子
rotate_kernel = np.array([
    [math.cos(theta), math.sin(theta)],
    [-math.sin(theta), math.cos(theta)]
])
for i in range(h):
    for j in range(w):
        origin = np.array([[i], [j]])
        x, y = np.matmul(rotate_kernel, origin).reshape(2, ).astype(np.int32)
        if 0 <= x <= h - 1 and 0 <= y <= w - 1:
            new_img[x, y] = img[i, j]
cv.imshow('Result', new_img)

cv.waitKey(0)
cv.destroyAllWindows()
