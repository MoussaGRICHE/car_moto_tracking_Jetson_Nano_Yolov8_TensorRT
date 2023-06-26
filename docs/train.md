
# Train Yolov8 model:

This project uses a camera embedded on the Jetson Nano to detect and track cars and motorcycles in a parking area. The project is developed using a deep learning model YOLOv8 and optimized for real-time performance in order to to improve parking lot management.


## Table of Contents:

    Data Preparation
    Model Development
    Testing and Evaluation
 

## Data Preparation:

To train the YOLOv8 model, you need a dataset of images or videos of the parking area. You can collect the dataset using the USB camera connected to the Jetson Nano. You may need to label the data to identify vehicles in the images. You can use pre-trained models and transfer learning techniques to reduce the amount of labeled data needed for your project.

In order  to train a model that works good in real context, you should take the following parameters into consideration when choosing the new images.
 
### 1- Parking type:
#### 1.1- underground parking
![1](https://user-images.githubusercontent.com/103992437/231197235-65f4802a-f5f3-4fdc-9de0-c161af29e587.JPG)

#### 1.2- outside parking
![2](https://user-images.githubusercontent.com/103992437/231197241-8940835e-08f4-402a-b7b0-56151f12250e.JPG)

### 2- Camera angle:
![3](https://user-images.githubusercontent.com/103992437/231197244-6254ada0-0e9e-48d8-a503-80eac7d016c7.JPG)

### 3- Camera distance:
![4](https://user-images.githubusercontent.com/103992437/231197245-8184b490-93cd-420c-9085-766b507fa334.JPG)

### 4- Outdoor weather:
#### 4.1- Sunny
![5](https://user-images.githubusercontent.com/103992437/231197249-a5276b08-55ee-41d2-b3f9-d40fda2aa49e.JPG)

#### 4.2- Rainy
![6](https://user-images.githubusercontent.com/103992437/231197254-d049eea9-7b36-49f2-b5cb-f2b76c131470.JPG)

#### 4.3- Overcast
![7](https://user-images.githubusercontent.com/103992437/231197258-502bf64c-3baf-44c4-86b7-d75deb186758.JPG)

Labeling thousands of images which contain multiple objects is a very time-consuming task. As a solution of this, you could follow the following work methodology that helps you to label automatically.


<img width=800 src="./assets/Diagramme-project.jpg" alt="Diagramme-project"> 

## Model Development:

YOLO (You Only Look Once) is a popular object detection model known for its speed and accuracy. It was first introduced by Joseph Redmon et al. in 2016 and has since undergone several iterations, the latest being YOLO v8. The Yolo models family are a single-stage real-time object detector that uses a fully convolutional neural network (CNN) to process an image. 

The latest version of Yolo is Yolov8 (January 2023). YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection, image segmentation and image classification tasks.
The Yolov8 models are presented in the following table:

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |

For purpose of deployment, you should train a model as small as possible. The model will be deployed in a limited resource environment and it will execute frames in real time. Indeed, the best model for this context seems the Yolov8n.

In order to get a small and a good model, we should train the model with images that look like images from the production environment (cars and motorcycles in parking lots) and generals images (cars and motorcycles in different contexts) to avoid the data drift problem.

### Global Dataset:

The global dataset used to train our model contains:

train:

•	The custom dataset: 2491 images

•	The augmented dataset: 4301 images

•	The general dataset: 7111 images

validation:

•	1280 images
    
test:

•	600 images


<img width=200 src="./assets/global-dataset.jpg" alt="global-dataset">

When training with your own data, you can use data as COCO_dataset, bdd100k_dataset and stanford_dataset or other sources and place it in the following directory.    
    
    [your data path]
    ┣ train        ┳ image001.jpg
    ┃              ┣ image001.txt
    ┃              ┣ image002.jpg
    ┃              ┣ image002.txt
    ┃              ┣ ・・・
    ┃              ┣ imageXXX.jpg
    ┃              ┗ imageXXX.txt
    ┣ validation   ┳ image_val001.jpg 
    ┃              ┣ image_val001.txt
    ┃              ┣ image_val002.jpg
    ┃              ┣ image_val0012.txt
    ┃              ┣ ・・・
    ┃              ┣ image_valXXX.jpg
    ┃              ┗ image_valXXX.txt 
    ┃ 
    ┗ test         ┳ image_test001.jpg 
                   ┣ image_test001.txt
                   ┣ image_test002.jpg
                   ┣ image_test0012.txt
                   ┣ ・・・
                   ┣ image_testXXX.jpg
                   ┗ image_testXXX.txt   
    
You can use the dataset_split.py to splitte you dataset into 85% train, 10% validation an 5% test.
    
### Important: 
You should add the dataset path to the configue file data.yaml
    
## Validation and Evaluation:

Evaluate the performance of your model on a validation dataset (images unseen by the model). Measure the mAP (mean Avrage Precision), and recall of your model. If your model is not meeting your objectives, iterate on the development process until you achieve the desired results.

For our model, the metrics of the trained YOLOv8n on Global_dataset are shown in the followin picture.


![results](https://github.com/MoussaGRICHE/car_moto_tracking_Jetson_Nano_Yolov8_TensorRT/assets/103992437/1d31c5b8-0dc2-46d5-8194-3629f98e355b)


![val_batch0_pred](https://github.com/MoussaGRICHE/car_moto_tracking_Jetson_Nano_Yolov8_TensorRT/assets/103992437/1ba3e2bc-1b2a-4c7e-9f3a-ff93eeeca13c)



