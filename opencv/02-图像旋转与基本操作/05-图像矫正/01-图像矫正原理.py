import cv2
import numpy as np

img = cv2.imread("./desk-pad_300x280.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# x, y, w, h = cv2.boundingRect(contours[0])
for contour in contours:
    for item in contour:
        print(item)
# cv2.rectangle(binary, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)

# cv2.imshow("binary", binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
