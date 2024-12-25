import cv2 as cv

# ----------------镜像-----------------------------------

img = cv.imread("../assets/example.png", 1)
rows, cols, channels = img.shape
# 设置镜像类型，0: 沿x轴镜像，>0: 沿y轴镜像，<0: 沿x轴和y轴都镜像
flip_type = 0
img = cv.flip(img, flip_type)
cv.imshow('img', img)

cv.waitKey(0)
cv.destroyAllWindows()
