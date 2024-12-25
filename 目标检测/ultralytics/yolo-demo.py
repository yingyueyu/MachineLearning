from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
# train_results = model.train(
#     data="banana-detection.yaml",  # path to dataset YAML 默认目录在datasets下
#     epochs=100,  # number of training epochs
#     imgsz=640,  # training image size
#     device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
# )
#
# # Evaluate model performance on the validation set
# metrics = model.val()

# Perform object detection on an image
results = model("./imgs")
results[0].show()
#
# # Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model
# print(path)