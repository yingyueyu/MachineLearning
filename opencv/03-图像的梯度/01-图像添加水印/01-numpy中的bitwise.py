import matplotlib.pyplot as plt
import numpy as np

img1 = np.zeros((5, 5, 3), dtype=np.uint8)
# img1 = np.ones((5, 5, 3), dtype=np.uint8) * 255
img1[0::3, 0::3] = (0, 255, 0)
img2 = np.zeros((5, 5, 3), dtype=np.uint8)
# img2 = np.ones((5, 5, 3), dtype=np.uint8) * 255
img2[1::2, 1::2] = (255, 0, 0)

plt.subplot(221)
plt.title("img1")
plt.axis("off")
plt.imshow(img1)
plt.subplot(222)
plt.title("img2")
plt.axis("off")
plt.imshow(img2)
plt.subplot(223)
plt.title("bitwise_and")
plt.axis("off")
plt.imshow(np.bitwise_and(img1, img2))
plt.subplot(224)
plt.axis("off")
plt.title("bitwise_or")
plt.imshow(np.bitwise_or(img1, img2))
plt.show()
