import cv2
import numpy as np

# 分别读取模板图像与原图
m_pic = cv2.imread("../assets/dog1.png", cv2.IMREAD_GRAYSCALE)
pic = cv2.imread("../assets/dogs.png", cv2.IMREAD_GRAYSCALE)
src = cv2.imread("../assets/dogs.png")
h, w = m_pic.shape

result = cv2.matchTemplate(pic, m_pic, cv2.TM_CCORR_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
x1, y1 = max_loc
cv2.rectangle(src, (x1, y1), (x1 + w, y1 + h), color=(0, 0, 255), thickness=2)
cv2.imshow("src", src)
cv2.waitKey(0)
