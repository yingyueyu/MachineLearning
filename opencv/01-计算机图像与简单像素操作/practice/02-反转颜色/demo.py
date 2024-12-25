import cv2
import numpy as np

img = cv2.imread("./assets/example_400x300.png")
h, w, channel = img.shape

blue_channel = img[:, :, 0].reshape(h, w, 1)
green_channel = img[:, :, 1].reshape(h, w, 1)
red_channel = img[:, :, 2].reshape(h, w, 1)
_fill = np.ones((h, w, 1), dtype=img.dtype) * 255
blue_channel = _fill - blue_channel
green_channel = _fill - green_channel
red_channel = _fill - red_channel


cv2.imshow("src", np.concatenate((blue_channel, green_channel, red_channel), axis=2))
cv2.waitKey(0)
cv2.destroyAllWindows()
