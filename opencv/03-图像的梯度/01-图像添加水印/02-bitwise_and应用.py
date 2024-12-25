import matplotlib.pyplot as plt
import numpy as np
import cv2

img1 = cv2.imread("./panel.png")
img2 = cv2.imread("./panel_mask.png")

img3 = np.bitwise_and(img1, img2)
cv2.imshow("img", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
