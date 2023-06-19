from ultralytics import YOLO

def train_yolo(yolo_model, epochs):
    """
    Function to train a YOLO model.
    Args:
        yolo_model (str): Path to the YOLO model file.
        epochs (int): Number of epochs to train the model.
    """
    # loading a pretrained model
    model = YOLO(yolo_model)

    # train the model
    model.train(data='./data.yaml', epochs=epochs, batch=6) 

def export_onnx(trained_yolo_model, format):
    """
    Function to export a trained YOLO model to ONNX format.
    Args:
        trained_yolo_model (str): Path to the trained YOLO model file.
        format (str): Format to export the model to (e.g., 'onnx').
    """
    # loading a pretrained model
    model = YOLO(trained_yolo_model)

    # export the model
    model.export(format=format) 

if __name__ == '__main__':
    # launch the model training
    train_yolo("yolov8n.pt", 2)

    # export the trained model to ONNX
    export_onnx("./runs/detect/train/weights/best.pt", "onnx")


