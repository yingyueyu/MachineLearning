import cv2

img = cv2.imread("../assets/example.png", cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
