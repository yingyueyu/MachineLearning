import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../assets/salt_pepper_noise_example2.png")
plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
filter_img = cv2.blur(img, (3, 3))
plt.subplot(122)
plt.imshow(cv2.cvtColor(filter_img, cv2.COLOR_BGR2RGB))
plt.show()
