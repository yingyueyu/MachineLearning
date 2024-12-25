import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../assets/scene.png", 0)
h, w = img.shape

# 直方图均衡化
his_equ = cv2.equalizeHist(img)
print(his_equ.shape)
print(np.hstack((img,his_equ)).shape)

plt.imshow(his_equ, cmap="gray")
plt.show()
