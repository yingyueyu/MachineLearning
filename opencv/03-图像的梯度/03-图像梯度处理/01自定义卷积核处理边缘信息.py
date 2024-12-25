import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../assets/number.png")
h, w, _ = img.shape
new_img = np.zeros((h, w), dtype=np.uint8)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 自定义卷积核
kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
kernel2 = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])
kernel3 = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

# 边界填充
gray = np.concatenate((gray[:, 1, None], gray, gray[:, -2, None]), axis=1)
gray = np.concatenate((gray[1, None, :], gray, gray[-2, None, :]), axis=0)

# 卷积运算
for i in range(h):
    for j in range(w):
        new_img[i, j] = np.sum(gray[i:i + 3, j:j + 3] * kernel2)
plt.subplot(121)
plt.imshow(gray, cmap="gray")
plt.subplot(122)
plt.imshow(new_img, cmap="gray")
plt.show()
