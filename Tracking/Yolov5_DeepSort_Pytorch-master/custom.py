

# import sys
# sys.path.insert(0, './yolov5')

from yolov5.utils.downloads import attempt_download
# from yolov5.utils.general import check_img_size
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import math
import warnings
warnings.simplefilter('ignore')

path = "..\\..\\data\\new.mp4"
deep_sort_weights = 'deep_sort_pytorch\\deep_sort\\deep\\checkpoint\\ckpt.t7'
config_deepsort="deep_sort_pytorch\\configs\\deep_sort.yaml"
font = cv2.FONT_HERSHEY_DUPLEX

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

# initialize deepsort
cfg = get_config()
cfg.merge_from_file(config_deepsort)
attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

print("loading.. yolo")    
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device = "cpu")




def x1y1x2y2_to_xywh(x1y1x2y2):
    xywhs = torch.zeros_like(x1y1x2y2)
    for i,xyxy in enumerate(x1y1x2y2):
        x1,y1,x2,y2 = xyxy[0],xyxy[1],xyxy[2],xyxy[3]
        w = x2 - x1
        h = y2 - y1
        xywhs[i,0] = x1
        xywhs[i,1] = y1
        xywhs[i,2] = w 
        xywhs[i,3] = h
    return xywhs



images = []
cap = cv2.VideoCapture(path)
while True:

    success, frame = cap.read()

    
    if not success:
        break

    imgsz = (frame.shape[1],frame.shape[0])
    print("frame size:",imgsz)
    # ideal_size = check_img_size(imgsz)
    # print("ideal size",ideal_size)
    # if imgsz != ideal_size:
    frame = cv2.resize(frame,(640,480), interpolation = cv2.INTER_AREA)
    # resize_factor = 1.3
    
    results = model(frame)


    det = results.xyxy[0]
    if det is not None and len(det):
        # det[:, :4] = scale_coords((640,352), det[:, :4], frame.shape).round()
        x1y1x2y2 = det[:,0:4]
        xywhs = x1y1x2y2_to_xywh(x1y1x2y2)

        confs = det[:,4]
        clss = det[:,5]
        
        outputs = deepsort.update(xywhs, confs, clss, frame)
        
        #draw boxes for visulization

        if len(outputs) > 0:
            for j, (output, conf) in enumerate(zip(outputs, confs)): 
                            
                bboxes = output[0:4]
                x1,y1,xc,yc = int(bboxes[0]),int(bboxes[1]),int(bboxes[2]),int(bboxes[3])
                id = output[4]
                cls = output[5]
                ###########
                h = yc - y1
                w = xc - x1
                x1 = int(xc - w/2)
                y1 = int(yc - h/2)
                x2 = int(xc + w/2)
                y2 = int(yc + h/2)
                     
                ###########
                c = int(cls)  # integer class
                label = f'{id} {names[c]} {conf:.2f}'
                frame = cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                frame = cv2.putText(frame,label,(x1-1,y1-1),font,0.5,(255,0,255),1)
            images.append(frame)
            cv2.imshow("Object detection",frame)
            cv2.waitKey(1)

    else:
         deepsort.increment_ages()

    
size = images[0].shape
out = cv2.VideoWriter('third_eye_tracker.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 30, (size[1],size[0]))
for i in range(len(images)):
    out.write(images[i])
out.release()

