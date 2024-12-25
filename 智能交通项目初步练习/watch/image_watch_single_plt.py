from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2

model = YOLO("../weights/yolo11n.pt")

results = model(["../data/test_img/test1.jpg"])

image = cv2.imread("../data/test_img/test1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

ax = plt.subplot(111)
ax.imshow(image)

for result in results:
    boxes = result.boxes
    cls = result.boxes.cls
    conf = result.boxes.conf
    for i, (x1, y1, x2, y2) in enumerate(boxes.xyxy):
        if int(cls[i].item()) not in [2, 5]:
            continue
        obj_conf = conf[i].item()
        x1 = x1.item()
        y1 = y1.item()
        x2 = x2.item()
        y2 = y2.item()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1), color="blue", fill=False)
        circle = patches.Circle((cx, cy), 2, color="blue", fill=True)
        ax.text(x1, y1, f"{obj_conf:.4f}", color="blue")
        ax.add_patch(rect)
        ax.add_patch(circle)

plt.show()
