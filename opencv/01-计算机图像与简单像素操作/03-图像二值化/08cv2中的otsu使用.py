import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../assets/example.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
_, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
_, thresh4 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
_, thresh5 = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)

titles = ['img', 'THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
cv2.waitKey(0)
