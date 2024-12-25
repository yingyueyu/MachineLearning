from ultralytics import YOLO

# Load a model
model = YOLO("weights/yolo11n.pt")

# 在可见的情况下，为每个能够一直追踪的物体设置追踪的id
results = model.track(source="视频链接", show=True, tracker="./trackers/bytetrack.yaml")
