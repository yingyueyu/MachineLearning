import cv2
import numpy as np

img = cv2.imread("../assets/example.png")
h, w, channel = img.shape

blue_channel = img[:, :, 0].reshape(h, w, 1)
green_channel = img[:, :, 1].reshape(h, w, 1)
red_channel = img[:, :, 2].reshape(h, w, 1)
_black = np.zeros(shape=(h, w, 1), dtype=img.dtype)

blue_img = np.concatenate((blue_channel, _black, _black), axis=2)
green_img = np.concatenate((_black, green_channel, _black), axis=2)
red_img = np.concatenate((_black, _black, red_channel), axis=2)

cv2.imshow("blue", blue_img)
cv2.imshow("green", green_img)
cv2.imshow("red", red_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imshow("src1", blue_img + green_img + red_img)
# cv2.imshow("src2", np.concatenate((blue_channel, green_channel, red_channel), axis=2))
