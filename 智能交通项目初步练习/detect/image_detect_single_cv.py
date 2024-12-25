import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2

model = YOLO("../weights/yolo11n.pt")

# 设置黑色掩膜区域
black1 = np.array([[[0, 0], [0, 530], [442, 192], [442, 0]]])
black2 = np.array([[[526, 0], [526, 192], [958, 482], [958, 0]]])
image = cv2.imread("../data/MVI_39511/img00001.jpg")
cv2.fillPoly(image, black1, color=(0, 0, 0))
cv2.fillPoly(image, black2, color=(0, 0, 0))
results = model(image)

for result in results:
    boxes = result.boxes
    cls = result.boxes.cls
    conf = result.boxes.conf
    for i, (x1, y1, x2, y2) in enumerate(boxes.xyxy):
        if int(cls[i].item()) not in [2, 5]:
            continue
        obj_conf = conf[i].item()
        x1 = int(x1.item())
        y1 = int(y1.item())
        x2 = int(x2.item())
        y2 = int(y2.item())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
        cv2.circle(image, (cx, cy), 2, color=(0, 255, 0), thickness=1)
        cv2.putText(image, f"{obj_conf:.4f}", (x1, y1), 0, 0.6, color=(0, 255, 0), thickness=1)

cv2.imshow("image", image)
cv2.waitKey(0)
