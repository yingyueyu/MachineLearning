from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2

CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors',
    'teddy bear', 'hair drier', 'toothbrush')

model = YOLO("weights/yolov8n.pt")

results = model(["test1.jpg"])

image = cv2.imread("test1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

ax = plt.subplot(111)
ax.imshow(image)

for result in results:
    boxes = result.boxes
    cls = result.boxes.cls
    print(cls)
    conf = result.boxes.conf
    for i, (x1, y1, x2, y2) in enumerate(boxes.xyxy):
        cls_name = CLASSES[int(cls[i].item())]
        obj_conf = conf[i].item()
        x1 = x1.item()
        y1 = y1.item()
        x2 = x2.item()
        y2 = y2.item()
        rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1), color="red", fill=False)
        ax.text(x1, y1,cls_name + f"{obj_conf:.4f}",color="red")
        ax.add_patch(rect)

plt.show()
