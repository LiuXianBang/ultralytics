from ultralytics import YOLO
from ultralytics.nn import DetectionModel

# Load a model
model = YOLO("ultralytics/cfg/models/11/yolo11n-MAFPN.yaml")
print(model.model.named_modules())
# print(model.info(detailed=True))
# DetectionModel("ultralytics/cfg/models/11/yolo11n-MAFPN.yaml")

# # Train the model
# train_results = model.train(
#     data="coco8.yaml",  # path to dataset YAML
#     epochs=100,  # number of training epochs
#     imgsz=640,  # training image size
#     device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
# )
#
# # Evaluate model performance on the validation set
# metrics = model.val()
#
# # Perform object detection on an image
# results = model("path/to/image.jpg")
# results[0].show()
#
# # Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model