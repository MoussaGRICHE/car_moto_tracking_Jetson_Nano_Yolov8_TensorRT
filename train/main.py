from io import BytesIO
import argparse
import onnx
import torch
from ultralytics import YOLO
try:
    import onnxsim
except ImportError:
    onnxsim = None
from typing import Tuple
import torch
import torch.nn as nn
from torch import Graph, Tensor, Value

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def make_anchors(feats: Tensor, strides: Tensor, grid_cell_offset: float = 0.5) -> Tuple[Tensor, Tensor]:
    """
    Function to create anchor points and stride tensors.
    Args:
        feats (Tensor): Input feature tensor.
        strides (Tensor): Stride tensor.
        grid_cell_offset (float): Grid cell offset value.
    Returns:
        Tuple[Tensor, Tensor]: Anchor points and stride tensor.
    """
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

class TRT_NMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Graph, boxes: Tensor, scores: Tensor, iou_threshold: float = 0.65,
                score_threshold: float = 0.25, max_output_boxes: int = 100, background_class: int = -1,
                box_coding: int = 0, plugin_version: str = '1', score_activation: int = 0) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass of the TRT_NMS function.
        Args:
            ctx (Graph): Graph context.
            boxes (Tensor): Boxes tensor.
            scores (Tensor): Scores tensor.
            iou_threshold (float): IoU threshold value.
            score_threshold (float): Score threshold value.
            max_output_boxes (int): Maximum number of output boxes.
            background_class (int): Background class value.
            box_coding (int): Box coding value.
            plugin_version (str): Plugin version value.
            score_activation (int): Score activation value.
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: Number of detections, boxes, scores, and labels.
        """
        batch_size, num_boxes, num_classes = scores.shape
        num_dets = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        boxes = torch.randn(batch_size, max_output_boxes, 4)
        scores = torch.randn(batch_size, max_output_boxes)
        labels = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_dets, boxes, scores, labels

    @staticmethod
    def symbolic(g, boxes: Value, scores: Value, iou_threshold: float = 0.45,
                 score_threshold: float = 0.25, max_output_boxes: int = 100, background_class: int = -1,
                 box_coding: int = 0, score_activation: int = 0, plugin_version: str = '1') -> Tuple[Value, Value, Value, Value]:
        """
        Symbolic implementation of the TRT_NMS function.
        Args:
            g (Graph): Graph object.
            boxes (Value): Boxes value.
            scores (Value): Scores value.
            iou_threshold (float): IoU threshold value.
            score_threshold (float): Score threshold value.
            max_output_boxes (int): Maximum number of output boxes.
            background_class (int): Background class value.
            box_coding (int): Box coding value.
            score_activation (int): Score activation value.
            plugin_version (str): Plugin version value.
        Returns:
            Tuple[Value, Value, Value, Value]: Number of detections, boxes, scores, and classes.
        """
        out = g.op('TRT::EfficientNMS_TRT', boxes, scores, iou_threshold_f=iou_threshold,
                   score_threshold_f=score_threshold, max_output_boxes_i=max_output_boxes,
                   background_class_i=background_class, box_coding_i=box_coding,
                   plugin_version_s=plugin_version, score_activation_i=score_activation, outputs=4)
        nums_dets, boxes, scores, classes = out
        return nums_dets, boxes, scores, classes

class PostDetect(nn.Module):
    export = True
    shape = None
    dynamic = False
    iou_thres = 0.65
    conf_thres = 0.25
    topk = 100

    def __init__(self, *args, **kwargs):
        """
        Post-processing detection module.
        """
        super().__init__()

    def forward(self, x):
        """
        Forward pass of the PostDetect module.
        Args:
            x: Input tensor.
        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: Number of detections, boxes, scores, and labels.
        """
        shape = x[0].shape
        b, res, b_reg_num = shape[0], [], self.reg_max * 4
        for i in range(self.nl):
            res.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        x = [i.view(b, self.no, -1) for i in res]
        y = torch.cat(x, 2)
        boxes, scores = y[:, :b_reg_num, ...], y[:, b_reg_num:, ...].sigmoid()
        boxes = boxes.view(b, 4, self.reg_max, -1).permute(0, 1, 3, 2)
        boxes = boxes.softmax(-1) @ torch.arange(self.reg_max).to(boxes)
        boxes0, boxes1 = -boxes[:, :2, ...], boxes[:, 2:, ...]
        boxes = self.anchors.repeat(b, 2, 1) + torch.cat([boxes0, boxes1], 1)
        boxes = boxes * self.strides

        return TRT_NMS.apply(boxes.transpose(1, 2), scores.transpose(1, 2), self.iou_thres, self.conf_thres, self.topk)

def optim(module: nn.Module):
    """
    Function to optimize the module.
    Args:
        module (nn.Module): PyTorch module.
    """
    s = str(type(module))[6:-2].split('.')[-1]
    if s == 'Detect':
        setattr(module, '__class__', PostDetect)

def export(args):
    """
    Function to export the trained model to ONNX format.
    Args:
        args: Arguments for the export process.
    """
    b = args.input_shape[0]
    YOLOv8 = YOLO(args.weights)
    model = YOLOv8.model.fuse().eval()
    for m in model.modules():
        optim(m)
        m.to(args.device)
    model.to(args.device)
    fake_input = torch.randn(args.input_shape).to(args.device)
    for _ in range(2):
        model(fake_input)
    save_path = args.weights.replace('.pt', '.onnx')
    with BytesIO() as f:
        torch.onnx.export(
            model,
            fake_input,
            f,
            opset_version=args.opset,
            input_names=['images'],
            output_names=['num_dets', 'bboxes', 'scores', 'labels'])
        f.seek(0)
        onnx_model = onnx.load(f)
    onnx.checker.check_model(onnx_model)
    shapes = [b, 1, b, args.topk, 4, b, args.topk, b, args.topk]
    for i in onnx_model.graph.output:
        for j in i.type.tensor_type.shape.dim:
            j.dim_param = str(shapes.pop(0))
    if args.sim:
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')
    onnx.save(onnx_model, save_path)
    print(f'ONNX export success, saved as {save_path}')




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


if __name__ == '__main__':
    # launch the model training
    train_yolo("yolov8n.pt", 300)

    # export the trained model to ONNX
    if torch.cuda.is_available()==True:
        device = "cuda"
    else:
        device = "cpu"
    args = argparse.Namespace(weights='./runs/detect/train/weights/best.pt', iou_thres=0.50, conf_thres=0.25, topk=100, opset=11, sim=False, input_shape=[1, 3, 640, 640], device=device)
    export(args)


