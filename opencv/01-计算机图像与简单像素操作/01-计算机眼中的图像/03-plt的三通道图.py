import cv2
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(10, 3))

img = cv2.imread("../assets/example.png")
blue_img = np.zeros(img.shape, dtype=img.dtype)
blue_img[:, :, 0] = img[:, :, 0]
blue_img = cv2.cvtColor(blue_img, cv2.COLOR_BGR2RGB)
ax1 = fig.add_subplot(131)
ax1.imshow(blue_img)
green_img = np.zeros(img.shape, dtype=img.dtype)
green_img[:, :, 1] = img[:, :, 1]
green_img = cv2.cvtColor(green_img, cv2.COLOR_BGR2RGB)
ax2 = fig.add_subplot(132)
ax2.imshow(green_img)
red_img = np.zeros(img.shape, dtype=img.dtype)
red_img[:, :, 2] = img[:, :, 2]
red_img = cv2.cvtColor(red_img, cv2.COLOR_BGR2RGB)
ax1 = fig.add_subplot(133)
ax1.imshow(red_img)

plt.show()
