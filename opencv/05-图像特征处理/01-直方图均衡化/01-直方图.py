import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../assets/scene.png", 0)
h, w = img.shape
plt.subplot(121)
plt.imshow(img, cmap="gray")

per_px_count = np.zeros((256,), dtype=np.int32)
for i in range(h):
    for j in range(w):
        per_px_count[img[i, j]] += 1

plt.subplot(122)
plt.bar(np.arange(0, 256), per_px_count)
plt.show()
