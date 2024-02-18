# Chapter2 YOLOv8 Implementation
## Running YOLOv8 on Windows
### Conda Environment
* Install yolo8
  ```bash
    conda create -n yolo8 python=3.9
    conda activate yolo8
	conda install -c conda-forge charset-normalizer
    pip install ultralytics==8
  ```
* To install pytorch, please visit the website: https://pytorch.org/get-started/locally/, select the solution and execute install cli.
### Test Image
* detect cli
```bash
yolo task=detect mode=predict model=yolov8n.pt source=image1.jpg
yolo task=detect mode=predict model=yolov8n.pt source=image1.jpg conf=0.8
yolo task=detect mode=predict model=yolov8n.pt source=image1.jpg save_txt=True
yolo task=detect mode=predict model=yolov8n.pt source=image1.jpg conf=0.8 save_txt=True save_crop=true
```
* segment cli
```bash
yolo task=segment mode=predict model=yolov8n-seg.pt source=image1.jpg
yolo task=segment mode=predict model=yolov8n-seg.pt source=image1.jpg hide_labels=True hide_conf=True
yolo task=segment mode=predict model=yolov8n-seg.pt source=image1.jpg conf=0.8
yolo task=segment mode=predict model=yolov8n-seg.pt source=image1.jpg save_txt=True
yolo task=segment mode=predict model=yolov8n-seg.pt source=image1.jpg conf=0.8 save_txt=True save_crop=true
yolo task=segment mode=predict model=yolov8n-seg.pt source=image1.jpg show=True
```

### Test Video
* cli
```bash
yolo task=detect mode=predict model=yolov8n.pt source=demo.mp4
yolo task=detect mode=predict model=yolov8n.pt source=demo.mp4  conf=0.8
```
### How the model convert to  ONNX (Open Neural Network Exchange) format ?
```bash
yolo task=detect mode=export model=yolov8n.pt  format=onnx 
```

## Running YOLOv8 in Google Colab
* https://colab.research.google.com/github/robert0714/Udemy-YOLOv8_Object_Detection_Tracking_Web_App_in_Python_2023/blob/main/chapter02/YOLOv8_Complete_Tutorial.ipynb
* https://colab.research.google.com/github/robert0714/Udemy-YOLOv8_Object_Detection_Tracking_Web_App_in_Python_2023/blob/main/chapter02/YOLOv8_Complete_Tutorial_Final.ipynb