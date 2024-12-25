import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../assets/morphologyEx3.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5, 5), np.uint8)

# 方式一：
result = cv2.dilate(gray, kernel)
result = cv2.erode(result, kernel)
# 方式二：
# result = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.axis("off")
plt.show()
