import numpy as np
import cv2 as cv


def resizeImage(image, width=None, height=None, inter=cv.INTER_AREA):
    newsize = (width, height)
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    # 高度算缩放比例
    if width is None:
        n = height / float(h)
        newsize = (int(n * w), height)
    else:
        n = width / float(w)
        newsize = (width, int(h * n))
    # 缩放图像
    newimage = cv.resize(image, newsize, interpolation=inter)
    return newimage


imageOriginal = cv.imread("../assets/example.png")
cv.imshow("Original", imageOriginal)
# 获取图像尺寸
w = width = imageOriginal.shape[0]
h = height = imageOriginal.shape[1]
print("Image Size:", w, h)
# 放大两倍
newimage = resizeImage(imageOriginal, w * 2, h * 2, cv.INTER_LINEAR)
cv.imshow("New", newimage)
# 保存缩放后的图像
cv.imwrite("newimage.jpg", newimage)
# 缩小五倍
newimage2 = resizeImage(imageOriginal, int(w / 5), int(h / 5), cv.INTER_LINEAR)
cv.imwrite("newimage2.jpg", newimage2)

cv.waitKey(0)
cv.destroyAllWindows()
