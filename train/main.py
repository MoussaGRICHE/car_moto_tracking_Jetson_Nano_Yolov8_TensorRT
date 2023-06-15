from ultralytics import YOLO



def train_yolo(yolo_model, epochs):
    """
    """
    # loading a pretrained model
    model = YOLO(yolo_model)

    # train the model
    model.train(data="./data.yaml", epochs=epochs, batch=6) 


def export_onnx(trained_yolo_model, format):
    """
    """
    # loading a pretrained model
    model = YOLO(trained_yolo_model)

    # train the model
    model.export(format=format) 

if __name__ == '__main__':
    # lunch the model training
    train_yolo("yolov8n.pt", 300)

    # export the trained model to onnx
    export_onnx("./train/runs/detect/train/weights/best.pt", onnx)


