import cv2
import numpy as np

img1 = cv2.imread("../assets/example.png")
img2 = cv2.imread("../assets/hqyj.png")
h, w, c = img2.shape

result = cv2.add(img1[:h, :w], img2)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
