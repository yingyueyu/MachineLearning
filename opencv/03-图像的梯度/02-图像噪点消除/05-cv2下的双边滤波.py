import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../assets/example.png")
plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
filter_img = cv2.bilateralFilter(img, 0, 100, 3,borderType=cv2.BORDER_REFLECT)

plt.subplot(122)
plt.imshow(cv2.cvtColor(filter_img, cv2.COLOR_BGR2RGB))
plt.show()
