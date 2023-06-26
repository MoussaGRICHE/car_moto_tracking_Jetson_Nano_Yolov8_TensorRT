# car_moto_tracking_Jetson_Nano_Yolov8_TensorRT
Detect, track and count cars and motorcycles using yolov8 and TensorRT on Jetson Nano


https://github.com/MoussaGRICHE/car_moto_tracking_Jetson_Nano_Yolov8_TensorRT/assets/103992437/fab1f0a6-9349-45bd-8e44-e923c8657b2a



https://github.com/MoussaGRICHE/car_moto_tracking_Jetson_Nano_Yolov8_TensorRT/assets/103992437/9879f396-1ac9-4389-b60a-42def786673c






To deploy this work on a Jetson Nano, you should do it in two steps:

## 1- On your PC:
### 1-1- Clone this repo on your PC:
On the car_moto_tracking_Jetson_Nano_Yolov8_TensorRT repository, create your own repository by clicking "Use this template" then "Create a new repository".

Put a name for your own repository then click "Create repository from template"

Then clone your own repository on your PC: 

	git clone "URL to your own repository"
	cd "repository_name"
	export root=${PWD}
	cd ${root}/train

### 1-2- Install the required packages:

	python -m pip install -r requirements.txt

### 1-3- Create a folder for the dataset:

	mkdir datasets
	cd datasets
	mkdir Train 
	mkdir Validation

### 1-4- Copy your dataset:
Copy your train dataset (images + .txt files) into datasets/Train

	cp -r path_to_your_train_dataset/. ${root}/train/datasets/Train

Copy your validation dataset (images + .txt files) into datasets/Validation

	cp -r path_to_your_validation_dataset/. ${root}/train/datasets/Validation


If you have more than car and motorcycle classes, modify the data.yaml to add the other classes.

### 1-5- Train the yolov8n model:

	python main.py

When the training is finished, your custom yolov8n model will be saved in 
${root}/train/runs/detect/train/weights/best.pt

### 1-6- Push your custom model to GitHub:

	git add runs/detect/train/weights/best.onnx
	git commit -am "Add the trained yolov8n model"
	git push

### Remark:

If you want to use an already pretrained model:

	python export_onnx.py --weights path_to_your_pretrained_model


	git add path_to_the_onnx_model
	git commit -am "Add the trained yolov8n model"
	git push
	
## 2- On the Jetson Nano:
### 2-1- Clone this repo on the Jetson Nano:

	git clone "URL to your own repository"
	cd "repository_name"
	export root=${PWD}

### 2-2- Export the engine from the onnx model

	/usr/src/tensorrt/bin/trtexec \
	--onnx=train/runs/detect/train/weights/best.onnx \
	--saveEngine=best.engine

After executing the above command, you will get an engine named best.engine .

### 2-3- For Detection:

	cd ${root}/detect
	mkdir build
	cd build
	cmake ..
	make
	cd ${root}

#### 2-3-1- Launch Detection:
for video:

	${root}/detect/build/yolov8_detect ${root}/best.engine video ${root}/src/test.mp4 1 show


#### Description of all arguments

- 1st argument : path to the maked file
- 2nd argument : path to the engine
- 3rd argument : video for saved video
- 4rth argument: path to video file
- 5th argument : if inference capacity of the Jetson is more then 30 fps, put 1, otherwise put 2, 3, 4 depending on the inference capacity of the Jetson
- 6th argument : show or save

for camera:

	${root}/detect/build/yolov8_detect ${root}/best.engine camera 1 show

#### Description of all arguments

- 1st argument : path to the maked file
- 2nd argument : path to the engine
- 3rd argument : camera for using embedded camera
- 4rth argument: if inference capacity of the Jetson is more then 30 fps, put 1, otherwise put 2, 3, 4 depending on the inference capacity of the Jetson
- 5th argument : show or save


### 2-4- For Tracking and Counting:

	cd ${root}/track_count
	mkdir build
	cd build
	cmake ..
	make
	cd ${root}

#### 2-4-1- Launch Tracking and Counting:
If you want to count only in one direction, put 1 as 7th argument. Otherwise, for 2 directions counting, put 2 as 7th argument.

Before displaying the processed video, the first frame of the video will be displayed. You should click on this frame to indicate the position of the line(s). For one direction counting, click twice and for 2 directions counting, click four time.

for video:

	${root}/track_count/build/yolov8_track_count ${root}/best.engine video ${root}/src/test.mp4 1 show

#### Description of all arguments

- 1st argument : path to the maked file
- 2nd argument : path to the engine
- 3rd argument : video for saved video
- 4rth argument: path to video file
- 5th argument : if inference capacity of the Jetson is more then 30 fps, put 1, otherwise put 2, 3, 4 depending on the inference capacity of the Jetson
- 6th argument : show or save

for camera:

	${root}/track_count/build/yolov8_track_count ${root}/best.engine camera 1 show

#### Description of all arguments

- 1st argument : path to the maked file
- 2nd argument : path to the engine
- 3rd argument : camera for using embedded camera
- 4rth argument: if inference capacity of the Jetson is more then 30 fps, put 1, otherwise put 2, 3, 4 depending on the inference capacity of the Jetson
- 5th argument : show or save

#### Remark:
If you are using the Jetson with SSH, you can not see the first frame of the video to draw the line. 

With SSH connection, follow these steps:

1- Add argument to the command line with ssh

	for video:

	${root}/track_count/build/yolov8_track_count ${root}/best.engine video ${root}/src/test.mp4 1 show ssh
	
	for camera
	
	${root}/track_count/build/yolov8_track_count ${root}/best.engine camera 1 show ssh
	
2- The first frame of the video will be saved

3- Copie this frame to you PC via SSH (in PC's terminal):

	scp jetson_name@jetson_server:path_to_frame_in_jetson/frame_for_line.jpg  ${root}/utils/frame_for_line.jpg
	
4- On your PC, launch the python script draw_line.py to draw the line and get the points.

	python3 ${root}/utils/draw_line.py
	
5- Give the points value to Jetson in the Jetson's terminal.
	
## References:

https://github.com/triple-Mu/YOLOv8-TensorRT

https://gitee.com/codemaxi-opensource/ByteTrack/tree/main

https://github.com/ultralytics/ultralytics


## Contributing:

Contributions are welcome! Please feel free to submit pull requests to improve the project.

## License:

This project is licensed under the MIT License - see the LICENSE.md file for details.


