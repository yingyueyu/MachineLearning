import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../assets/landscape.png", 0)
h, w = img.shape
plt.subplot(121)
plt.imshow(img, cmap="gray")

# 直方图均衡化
his_equ = cv2.equalizeHist(img)

plt.subplot(122)
plt.imshow(his_equ, cmap="gray")
plt.show()
