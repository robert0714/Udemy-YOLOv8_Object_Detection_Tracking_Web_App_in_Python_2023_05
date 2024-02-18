from ultralytics import YOLO

#Initialize YOLO with the Model Name
model = YOLO("yolov8n.pt")

##Predict Method Takes all the parameters of the Command Line Interface
model.predict(source='image1.jpg', save=True, conf=0.5, save_txt=True)

#model.predict(source='image1.jpg', save=True, conf=0.5, save_txt=True)
#model.export(format="onnx")