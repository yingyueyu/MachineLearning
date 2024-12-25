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
# 卷积运算
new_img = cv2.filter2D(img, -1, kernel)

plt.subplot(121)
plt.imshow(gray, cmap="gray")
plt.subplot(122)
plt.imshow(new_img, cmap="gray")
plt.show()
