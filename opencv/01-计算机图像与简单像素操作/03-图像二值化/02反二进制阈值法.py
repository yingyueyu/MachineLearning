import cv2

img = cv2.imread("../assets/example.png", cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
threshold = 127

for i in range(h):
    for j in range(w):
        if gray[i, j] > 120:
            gray[i, j] = 0
        else:
            gray[i, j] = 255

cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
