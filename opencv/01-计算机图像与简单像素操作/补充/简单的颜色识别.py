import cv2
import numpy as np

img = cv2.imread("./assets/basic01.png").copy()

h, w, channel = img.shape
blue_channel = img[:, :, 0]
green_channel = img[:, :, 1]
red_channel = img[:, :, 2]


def channel_mask(img_channel):
    h, w = img_channel.shape
    for i in range(h):
        for j in range(w):
            if 255 > img_channel[i, j] > 100:
                img_channel[i, j] = 255
            else:
                img_channel[i, j] = 0
    return img_channel


blue_mask = channel_mask(blue_channel.copy())
green_mask = channel_mask(green_channel.copy())
red_mask = channel_mask(red_channel.copy())

# 形态变换
kernel = np.ones((5, 5), np.uint8)
blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

# 找出颜色区域
mask = blue_mask + green_mask + red_mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    color_val = (255, 255, 255)
    if np.any(red_mask[y:y + h, x:x + w]):
        color_val = (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color=color_val)
        cv2.putText(img, "red", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color=color_val, thickness=2)
    if np.any(green_mask[y:y + h, x:x + w]):
        color_val = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color=color_val)
        cv2.putText(img, "green", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color=color_val, thickness=2)
    if np.any(blue_mask[y:y + h, x:x + w]):
        color_val = (255, 0, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color=color_val)
        cv2.putText(img, "blue", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, color=color_val, thickness=2)

cv2.imshow("test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
