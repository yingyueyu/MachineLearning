# 仿射变换
import cv2
import numpy as np

img = cv2.imread("../assets/perspective.png")

# 此处省略梯形坐标值的计算
pts1 = np.float32([
    [60,150],
    [240,150],
    [0,300],
    [300,300]
])

pst2 = np.float32([
    [0,0],
    [300,0],
    [0,300],
    [300,300]
])

# 仿射变化的核心
M = cv2.getPerspectiveTransform(pts1,pst2)

# 将梯形拉伸成了矩形
dic_img = cv2.warpPerspective(img,M,img.shape[:2])

cv2.imshow("dic",dic_img)
cv2.waitKey(0)