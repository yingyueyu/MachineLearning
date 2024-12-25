import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../assets/example.png")
plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
filter_img = np.zeros(img.shape, dtype=np.uint8)
h, w, _ = img.shape
# 边缘填充
img = np.concatenate((img[:, 2, None], img[:, 1, None], img, img[:, -2, None], img[:, -3, None]), axis=1)
img = np.concatenate((img[2, None, :], img[1, None, :], img, img[-2, None, :], img[-3, None, :]), axis=0)

sigmaColor = 200
sigmaSpace = 3
# 空域信息
sigma_s = np.zeros((5, 5))
m_x, m_y = 2, 2
for i in range(5):
    for j in range(5):
        a = ((i - m_y) ** 2 + (j - m_x) ** 2) / (2 * sigmaSpace ** 2)
        sigma_s[i][j] = np.exp(-a)


def bilateral_result(block):
    m = block[m_y, m_x]
    b = ((block - m) ** 2)/ (2 * sigmaColor ** 2)
    sigma_r = np.exp(-b)
    w_p = sigma_r * sigma_s
    return (np.sum(w_p * block) / np.sum(w_p)).astype(np.uint8)


# 值域信息
for i in range(h):
    for j in range(w):
        b_block = img[i:i + 5, j:j + 5, 0]
        g_block = img[i:i + 5, j:j + 5, 1]
        r_block = img[i:i + 5, j:j + 5, 2]
        b = bilateral_result(b_block)
        g = bilateral_result(g_block)
        r = bilateral_result(r_block)
        filter_img[i, j] = (b, g, r)

plt.subplot(122)
plt.imshow(cv2.cvtColor(filter_img, cv2.COLOR_BGR2RGB))
plt.show()
