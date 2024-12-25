import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("test.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.dtype)
_, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
h, w = binary_img.shape
a = binary_img.copy() / 255
a = np.concatenate((a[:, 0, None], a[:, 0, None], a, a[:, 1, None], a[:, 1, None]), axis=1)
a = np.concatenate((a[0, None, :], a[0, None, :], a, a[1, None, :], a[1, None, :]), axis=0)

kernel = [
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
]
kernel = np.array(kernel)

result = binary_img.copy()
for i in range(h):
    for j in range(w):
        part = a[i:i + 5, j:j + 5]
        result[i, j] = np.max(part * kernel)

result = result * 255
result = result.astype(np.uint8)

plt.subplot(1,2,1)
plt.imshow(binary_img,cmap="gray")
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(result,cmap="gray")
plt.axis("off")
plt.show()
