import cv2
import numpy as np

"""
形态学操作：
cv2.MORPH_ERODE：腐蚀操作。
cv2.MORPH_DILATE：膨胀操作。
cv2.MORPH_OPEN：开运算。
cv2.MORPH_CLOSE：闭运算。
cv2.MORPH_GRADIENT：形态学梯度。
cv2.MORPH_TOPHAT：原图像减去膨胀的图像。
cv2.MORPH_HITMISS：结构元素对应的点集比较。
"""

img = cv2.imread("../assets/morphologyEx3.png", 0)

kernel = np.ones((5, 5), np.uint8)
# 腐蚀操作
# eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
# 膨胀操作
# dilate = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
# 开运算
# open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# 闭运算
close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow("src",img)
cv2.imshow("erode",close)
cv2.waitKey(0)
cv2.destroyAllWindows()
