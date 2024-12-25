import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../assets/example.png")
plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.subplot(122)
plt.imshow(img)
plt.show()
