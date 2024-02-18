# how we can train Yolo V8 on custom dataset of potholes
## Kaggle Dataset
* https://www.kaggle.com/datasets/sachinpatel21/pothole-image-dataset
## we need to annotate the dataset
* https://app.roboflow.com/
### Potholes Detection
* We can find "pothole roads" in youtube .
* We can use public dataset annotated:
  * https://universe.roboflow.com/working-pfnxt/pothole-detection-project-bayaq/dataset/1#
* The model best weights: model={HOME}/runs/detect/train/weights/best.pt
* sample: https://drive.google.com/file/d/1iMitK9VCUWmBcZiiEPHK1d2pydALof6s
* Running YOLOv8 in Google Colab
  * https://colab.research.google.com/github/robert0714/Udemy-YOLOv8_Object_Detection_Tracking_Web_App_in_Python_2023_05/blob/main/chapter03/Potholes_Detection_Step_by_Step_Complete.ipynb
### Personal Protective Equipment Detection  
* We can use public dataset annotated:
  * https://universe.roboflow.com/objet-detect-yolov5/eep_detection-u9bbd/dataset/1#
* sample: 
  1. https://drive.google.com/file/d/1crFwrpMF1OlaJ0ZCZjBNRo9llLEVR8VQ
  2. https://drive.google.com/file/d/
  3. https://drive.google.com/file/d/1cTIBNQ1R_7JAOURVv9cJ6P935ym_IkZ0
  4. https://drive.google.com/file/d/1256pNK0nQnEDT6FRLQAraTRkOY7BSprq
* Running YOLOv8 in Google Colab
  * https://colab.research.google.com/github/robert0714/Udemy-YOLOv8_Object_Detection_Tracking_Web_App_in_Python_2023_05/blob/main/chapter03/PPE/PPE_Detection__Step_by_Step_Complete_Final.ipynb
### Pen and Book Detection
* Open Images Dataset v7: https://storage.googleapis.com/openimages/web/index.html
* Open Images Dataset v4: https://github.com/EscVM/OIDv4_ToolKit
* Using [roboflow](https://app.roboflow.com/) to convert to other format dataset
* Running YOLOv8 in Google Colab
  * https://colab.research.google.com/github/robert0714/Udemy-YOLOv8_Object_Detection_Tracking_Web_App_in_Python_2023_05/blob/main/chapter03/PenBook/Pen_Book_Detection_YOLOv8_Step_by_Step_Complete.ipynb
