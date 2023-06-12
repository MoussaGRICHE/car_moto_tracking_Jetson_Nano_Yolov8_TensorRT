# car_moto_tracking_Jetson_Nano_Yolov8_TensorRT
Detect, track and count cars and motorcycles using yolov8 and TensorRT in Jetson Nano


To diploye this work on a Jetson Nano, you should do it in two steps:

## 1- On your PC:
### 1-1- Clone this repo on your PC:
On car_moto_tracking_Jetson_Nano_Yolov8_TensorRT repository, create your own repositoty by clicking "Use this template" then "Create a new repository".

Put a name for your awn repository then click "Create repository from template"

Then clone your own repository on your PC: 

	git clone "URL to your own repository"
	cd "repository_name"
	export root=${PWD}
	cd ${root}/train

### 1-2- Install the requirements packages:

	python -m pip install -r requirements.txt

### 1-3- Create folder for the dataset:

	mkdir data
	cd data
	mkdir Train & validation

### 1-4- Copie your dataset:
copie you train dataset (images + .txt files) into data/Train

	cp -r path_to_your_train_datset/. ${root}/train/data/Train

copie you validation dataset (images + .txt files) into data/Validation

	cp -r path_to_your_validation_datset/. ${root}/train/data/Validation


If you have more then car and motorcycle classes, modify the data.yaml to add the other classes.

### 1-5- Train the yolov8n model:

	python train.py

When the training is finish, your custon yolov8n model will be saved in 
car_moto_tracking_Jetson_Nano_Yolov8_TensorRT/train/run/train/weights/best.pt

### 1-6- Push your custon model to github:

	git add run/train/weights/best.onnx
	git commit -am "Add the trained yolov8n model"
	git push




## 2- On the Jetson Nano:
### 2-1- Clone this repo on the Jetson Nano:

	git clone "URL to your own repository"
	cd "repository_name"
	export root=${PWD}

### 2-2- Convert

	/usr/src/tensorrt/bin/trtexec \
	--onnx= train/run/train/weights/best.onnx \
	--saveEngine=best.engine

After executing the above command, you will get an engine named best.engine .

### 2-2- For the Detection:

	cd ${root}/detect
	mkdir build
	cd build
	cmake ..
	make
	cd ${root}

#### 2-2-1- Launch the Detection:
for video:

	${root}/detect/build/yolov8_detect ${root}/best.engine video ${root}/src:test.mp4 	1 show

for camera:

	${root}/detect/build/yolov8_detect ${root}/best.engine camera 1 show

### 2-3- For the Tracking and Counting:

	cd ${root}/track_count
	mkdir build
	cd build
	cmake ..
	make
	cd ${root}

#### 2-3-1- Launch the Tracking and Counting:
for video:

	${root}/detect/build/yolov8_detect ${root}/best.engine video ${root}/src:test.mp4 	1 show 1

for camera:

	${root}/detect/build/yolov8_detect ${root}/best.engine camera 1 show 1
