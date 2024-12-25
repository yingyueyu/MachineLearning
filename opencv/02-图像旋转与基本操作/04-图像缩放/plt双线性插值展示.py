import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("./img.png")
h, w, c = img.shape
size = 50
new_img = np.zeros((size, size, 3), dtype=np.uint8)
for i in range(size):
    for j in range(size):
        # 一般的线性关系
        # new_img[i, j] = img[round(h / size * i), round(w / size * j)]
        # 居中的线性
        SrcX = (i + 0.5) * (w / size) - 0.5
        SrcY = (j + 0.5) * (h / size) - 0.5
        new_img[i, j] = img[round(SrcX), round(SrcY)]

plt.subplot(121)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.subplot(122)
plt.imshow(cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB))
plt.show()
