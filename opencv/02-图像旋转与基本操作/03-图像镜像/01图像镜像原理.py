import cv2 as cv

# ----------------镜像原理--------------------------------

img = cv.imread("../assets/example.png", 1)
rows, cols, channels = img.shape
# 设置镜像类型，0: 沿x轴镜像，>0: 沿y轴镜像，<0: 沿x轴和y轴都镜像
img_flip_x = img[::-1, ::-1, :]
cv.imshow('img', img_flip_x)

cv.waitKey(0)
cv.destroyAllWindows()
