import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("test.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.dtype)
_, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)

# 方式一：
# result = cv2.dilate(binary_img, kernel)
# 方式二：
result = cv2.morphologyEx(binary_img, cv2.MORPH_DILATE, kernel)

plt.subplot(1, 2, 1)
plt.imshow(binary_img, cmap="gray")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.show()
