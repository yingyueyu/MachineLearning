import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../assets/scene.png", 0)
h, w = img.shape
dist = img.copy()
plt.subplot(121)
plt.imshow(img, cmap="gray")

per_px_count = np.zeros((256,), dtype=np.int32)
for i in range(h):
    for j in range(w):
        per_px_count[img[i, j]] += 1

# 直方图均衡化
histogram_equalization = np.zeros((256,), dtype=np.int32)
for i in range(256):
    for j in range(i + 1):
        histogram_equalization[i] += per_px_count[j]

histogram_equalization = histogram_equalization * 255 / histogram_equalization[-1]
for i in range(h):
    for j in range(w):
        for k in range(256):
            if img[i, j] == k:
                dist[i, j] = histogram_equalization[k]
        # per_px_count[img[i, j]] += 1

plt.subplot(122)
plt.imshow(dist, cmap="gray")
# plt.bar(np.arange(0, 256), histogram_equalization)
plt.show()
