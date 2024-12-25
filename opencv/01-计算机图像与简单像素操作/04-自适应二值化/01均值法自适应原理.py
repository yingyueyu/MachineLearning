import cv2
import numpy as np

img = cv2.imread("../assets/book_page.png", cv2.IMREAD_UNCHANGED)
a = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = a.copy()
h, w = a.shape
a = np.concatenate((a[:, 0, None], a, a[:, -1, None]), axis=1)
a = np.concatenate((a[0, None, :], a, a[-1, None, :]), axis=0)

kernel = np.array([
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9],
    [1 / 9, 1 / 9, 1 / 9]
])

print(kernel)

for i in range(h):
    for j in range(w):
        threshold = np.sum(a[i:i + 3, j:j + 3] * kernel).astype(np.uint8)
        if gray[i, j] >= threshold:
            gray[i, j] = 255
        else:
            gray[i, j] = 0

cv2.imshow("img", gray)
cv2.waitKey(0)
