import cv2
img = cv2.imread("../assets/example.png",cv2.IMREAD_UNCHANGED)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

