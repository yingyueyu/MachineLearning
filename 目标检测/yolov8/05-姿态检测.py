from ultralytics import YOLO

model = YOLO("weights/yolo11n-pose.pt")

results = model("https://ultralytics.com/images/bus.jpg")
results[0].show()