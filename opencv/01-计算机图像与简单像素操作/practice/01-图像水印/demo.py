import cv2

src = cv2.imread("./assets/example_400x300.png")
log = cv2.imread("./assets/hqyj_pc_logo.png")

log_h, log_w, _ = log.shape

for i in range(log_h):
    for j in range(log_w):
        if log[i, j, 0] < 250 and log[i, j, 1] < 250 and log[i, j, 2] < 250:
            src[i, j] = log[i, j]

cv2.imshow("src", src)
cv2.waitKey(0)
cv2.destroyAllWindows()
