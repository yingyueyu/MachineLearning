import cv2

img = cv2.imread("../assets/example.png", cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

threshold = 10
dict_threshold = {}
while threshold < 255:
    fore_px_list = gray[gray >= threshold]
    back_px_list = gray[gray < threshold]
    n_0 = len(fore_px_list)
    n_1 = len(back_px_list)
    if n_0 == 0 or n_1 == 0:
        break
    w_0 = n_0 / (n_0 + n_1)
    w_1 = n_1 / (n_0 + n_1)
    u_0 = fore_px_list.mean()
    u_1 = back_px_list.mean()
    u = gray.mean()
    g = w_0 * (u_0 - u) ** 2 + w_1 * (u_1 - u) ** 2
    dict_threshold[f"{threshold}"] = g
    threshold += 1

result = sorted(dict_threshold, reverse=True)
best_threshold = int(result[0])

h, w = gray.shape
# 二进制阈值法
for i in range(h):
    for j in range(w):
        if gray[i, j] > best_threshold:
            gray[i, j] = 255
        else:
            gray[i, j] = 0

cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
