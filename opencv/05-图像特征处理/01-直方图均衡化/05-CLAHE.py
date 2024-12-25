# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../assets/test3.png')
clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img[:, :, 0])
cl2 = clahe.apply(img[:, :, 1])
cl3 = clahe.apply(img[:, :, 2])
cv2.imshow("img", img)
cv2.imshow("cl", cv2.merge((cl1, cl2, cl3)))
cv2.waitKey(0)
# res = np.hstack((img, cv2.merge(cl3,cl2,cl1)))
# plt.imshow(res)
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()
