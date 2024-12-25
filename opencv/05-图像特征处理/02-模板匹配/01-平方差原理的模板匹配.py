import cv2
import numpy as np

# 分别读取模板图像与原图
m_pic = cv2.imread("../assets/dog1.png", cv2.IMREAD_GRAYSCALE)
pic = cv2.imread("../assets/dogs.png", cv2.IMREAD_GRAYSCALE)
src = cv2.imread("../assets/dogs.png")
h, w = m_pic.shape
# 横向与纵向分别移动的次数
n = pic.shape[0] - h + 1
m = pic.shape[1] - w + 1

# 平方差记录列表 五位数:[x1,y1,x2,y2,sd]
records = []
# 进行平方差计算，并获取里面的较小的值（匹配度越好）
for i in range(n):
    for j in range(m):
        sd = np.sum((pic[i:i + h, j:j + w] - m_pic) ** 2)
        records.append([i, j, i + h, j + w, sd])

# 取出最小值，并画出来
best = sorted(records, key=lambda x: x[4])[0]
x1 = best[1]
x2 = best[3]
y1 = best[0]
y2 = best[2]
cv2.rectangle(src, [x1, y1], [x2, y2], color=(0, 0, 255), thickness=2)
cv2.imshow("src", src)
cv2.waitKey(0)
