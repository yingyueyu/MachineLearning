import cv2
import numpy as np

src = cv2.imread("../assets/signal.png")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(contours[0])
# 绘制方式一
# cv2.rectangle(src, (x, y), (x + w, y + h), color=(255, 255, 255), thickness=2)
# 绘制方式二
# points = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])
# cv2.drawContours(src, [points], 0, (255, 255, 255), 2)
# 绘制方式三
rect = cv2.minAreaRect(contours[0])
points = cv2.boxPoints(rect)
points = points.astype(np.uint)
cv2.drawContours(src, [points], 0, (255, 255, 255), 2)
cv2.imshow("result", src)
cv2.waitKey(0)
cv2.destroyAllWindows()
