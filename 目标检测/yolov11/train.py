from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# workers=0 指定训练时候的并发线程的数量，workers > 0 的时候可以在多GPU下使用
results = model.train(data="banana.yaml", epochs=100, imgsz=640, workers=0)
