
import os
import sys
from pathlib import Path

import numpy as np
import torch
import cv2

from utils.augmentations import letterbox

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import check_img_size, cv2, non_max_suppression, scale_coords


class Yolo():
    def __init__(self):
        
        self.settings = {
            "weights"  : 'weights/yolov5s.pt' ,
            "source"  : '0' ,
            "data": 'data/coco128.yaml',  
            "imgsz" :   (640, 640) ,
            "conf_thres"    :   0.25,  
            "iou_thres" :   0.45,  
            "max_det":1000,  
            "device":torch.device('cuda:0'),  
            "view_img":False,  
            "save_txt":False,  
            "save_conf":False,  
            "save_crop":False,  
            "nosave":False,  
            "classes":None,  
            "agnostic_nms":False,  
            "augment":False,  
            "visualize":False,  
            "update":False,  
            "project":'runs/detect',  
            "name":'exp',  
            "exist_ok":False,  
            "line_thickness":3,  
            "hide_labels":False,  
            "hide_conf":False,  
            "half":False,  
            "dnn":False  
        }
        self.device = self.settings["device"]
        self.model = DetectMultiBackend(self.settings["weights"], device= self.device, dnn=False, data=self.settings["data"], fp16=False)
        self.stride = self.model.stride
        self.imgsz=(640, 640)
        self.auto = True

    def preprocess(self, frame):
        imgsz = check_img_size(self.imgsz, s=self.stride)
        img = letterbox(frame, imgsz, stride=self.stride, auto=self.auto)[0]

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        return im

    def postprocess(self, im, im0, pred, classes):
        bboxes=[]
        pred = non_max_suppression(pred, self.settings["conf_thres"], self.settings["iou_thres"], classes, self.settings["agnostic_nms"], max_det=self.settings["max_det"])
        
        for i, det in enumerate(pred):  # per image
            
            gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                for *box, class_score, cls in reversed(det.detach().cpu().numpy()):
                    bboxes.append([box[0], box[1], box[2], box[3]])

        return bboxes

    def draw(self, canvas, bboxes):
        h, w, c = canvas.shape
        for bbox in bboxes:
            p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            canvas = cv2.rectangle(canvas, p1, p2, (255, 255, 255), 2)

        return canvas

    def detect(self, frame, classes):

        #! Ön işleme
        im = self.preprocess(frame)
        
        #! Inference
        result = self.model(im, augment=False)
        
        #! Son İşleme
        bboxes = self.postprocess(im, frame, result, classes)
        return bboxes

        #! Çizdirme
        # canvas=self.draw(frame, bboxes)

        # cv2.imshow('',canvas)
        # cv2.waitKey(0)


if __name__=="__main__":
    manager=Yolo()



    img = cv2.imread("data/images/zidane.jpg")
    bboxes = manager.detect(img, [0])

    manager.draw()
