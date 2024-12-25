import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../assets/salt_pepper_noise_example2.png")
plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
filter_img = np.zeros(img.shape, dtype=np.uint8)
h, w, _ = img.shape
# 边缘填充
img = np.concatenate((img[:, 1, None], img, img[:, -2, None]), axis=1)
img = np.concatenate((img[1, None, :], img, img[-2, None, :]), axis=0)

for i in range(h):
    for j in range(w):
        k = img[i:i + 3, j:j + 3]
        r = sorted(k[:, :, 0].reshape(-1, ))[4]
        g = sorted(k[:, :, 1].reshape(-1, ))[4]
        b = sorted(k[:, :, 2].reshape(-1, ))[4]
        filter_img[i, j] = (r, g, b)

plt.subplot(122)
plt.imshow(cv2.cvtColor(filter_img, cv2.COLOR_BGR2RGB))
plt.show()
