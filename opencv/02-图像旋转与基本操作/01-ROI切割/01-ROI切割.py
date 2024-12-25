import cv2
import numpy as np

# 读取图像
image = cv2.imread('../assets/example.png')
# 显示原始图像
cv2.imshow('Original Image', image)
# 等待用户选择ROI
roi = cv2.selectROI('ROI Selector', image, False)
# roi是一个包含左上角和右下角坐标的元组
x, y, w, h = roi
# 裁剪ROI
roi_image = image[y:y+h, x:x+w]
# 显示ROI
cv2.imshow('ROI', roi_image)
# 等待按键
cv2.waitKey(0)
# 关闭所有窗口
cv2.destroyAllWindows()
