import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../assets/number.png")
h, w, _ = img.shape
new_img = np.zeros((h, w), dtype=np.uint8)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 卷积运算
new_img = cv2.Laplacian(img,-1,ksize=3)

plt.subplot(121)
plt.imshow(gray, cmap="gray")
plt.subplot(122)
plt.imshow(new_img, cmap="gray")
plt.show()
