
from numpy.lib.function_base import angle
from yolov5.utils.downloads import attempt_download
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import cv2
import torch
import warnings
warnings.filterwarnings("ignore")
from PIL import Image,ImageOps
import numpy as np
from collections import deque
import math


## Setting parameters and variables##
path = "../data/new.mp4"
deep_sort_weights = 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
config_deepsort="deep_sort_pytorch/configs/deep_sort.yaml"
font = cv2.FONT_HERSHEY_DUPLEX
obstacles = ['car', 'person', 'motorcycle', 'train', 'truck','bicycle']
roi = 0.25
success = True
threshold = 0.3
size_threshold = 60
images = []
dim = (640, 480)
database = {}
counter = 0
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']


def new_obs(label):
    return {
        "c_hist":deque(maxlen=2), 
        "size_hist":deque(maxlen=2), 
        "angle":0, 
        "large":False, 
        "label":label
    }


def track_obs(database, id, xc, yc, frame):
    database[id]["c_hist"].append((int(xc),int(yc)))
    c_hist = database[id]["c_hist"]
    if len(c_hist) == 2:
        lenab = math.sqrt((c_hist[0][0]-c_hist[1][0])**2+(c_hist[0][1]-c_hist[1][1])**2)
        length = 100
        cx = c_hist[1][0] + (c_hist[1][0]-c_hist[0][0]) / (lenab + 0.001) *length
        cy = c_hist[1][1] + (c_hist[1][1]-c_hist[0][1]) / (lenab+ 0.001)  *length 
                 
        frame = cv2.arrowedLine(frame, c_hist[0], (int(cx),int(cy)),(123,232,324), 2)
    return database, frame

def draw_boxes(database, frame, outputs, confs, left, right, obs):
    mid = int((left+right)/2)
    if len(outputs) > 0:
        for j, (output, conf) in enumerate(zip(outputs, confs)):  
            label = names[int(output[5])]  # integer class
            if conf > threshold and label in obstacles:                   
                bboxes = output[0:4]
                x1,y1,x2,y2 = int(bboxes[0]),int(bboxes[1]),int(bboxes[2]),int(bboxes[3])
                id = output[4]
                xc = x1+(x2-x1)/2 # center-x
                yc = y1 + (y2-y1)/2 # center-y
                area = int(((x2-x1) * (y2-y1))/100)
                color = (0,255,0)
                
                ## Creating obs in database
                if id not in database.keys():
                    database[id] = new_obs(label)

                ## Trajectory ##
                database, frame = track_obs(database, id, xc, yc, frame)

                ## Based on size and location ##
                if x2>=left and x1<=right:
                    if area < size_threshold:
                        color = (0,255,255)
                    else:
                        color = (0,0,255)
                        if x2 <= mid:
                            obs[0] += 1
                        elif x1 >= mid:
                            obs[1] += 1
                        else:
                            obs[2] += 1
                        
                label = f'{id} {area} {conf:.2f}'
                frame = cv2.rectangle(frame, (x1,y1),(x2,y2),color,2)
                frame = cv2.putText(frame,label,(x1-1,y1-1),font,0.5,(255,0,255),1)
                

    ## Warning ##
    for i in range(3):
        if obs[i] >= 2:
            warn[i] += "WARNING"
    frame = cv2.putText(frame,str(obs[0]) + warn[0],(left,frame.shape[:2][0]),font,0.5,(0,0,255),2)
    frame = cv2.putText(frame,str(obs[2]) + warn[2],(mid,frame.shape[:2][0]),font,0.5,(0,0,255),2)
    frame = cv2.putText(frame,str(obs[1]) + warn[1],(right,frame.shape[:2][0]),font,0.5,(0,0,255),2)
    return database, frame

def draw_ROI(box_img,roi):
    height, width= frame.shape[:2]  
    left, right = int(roi * width), int((1-roi) * width)
    mid = int(width / 2)
    ROI_region = [[(left,height),(left,0),(right,0),(right,height)]]
    ROI_region2 = [[(left,height),(left,0),(mid,0),(mid,height)]]
    box_img = cv2.rectangle(box_img, ROI_region[0][1],ROI_region[0][3],(0,0,0),1)
    box_img = cv2.rectangle(box_img, ROI_region2[0][1],ROI_region2[0][3],(0,0,0),1)
    return box_img, left, right

def x1y1x2y2_to_xywh(x1y1x2y2):
    xywhs = torch.zeros_like(x1y1x2y2)
    for i,xyxy in enumerate(x1y1x2y2):
        x1,y1,x2,y2 = xyxy[0],xyxy[1],xyxy[2],xyxy[3]
        w = x2 - x1
        h = y2 - y1
        xc = x1 + w/2
        yc = y1 + h/2
        xywhs[i,0] = xc
        xywhs[i,1] = yc
        xywhs[i,2] = w 
        xywhs[i,3] = h
    return xywhs

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return np.asarray(ImageOps.expand(img, padding))

def save_output(images, fps):
    size = images[0].shape
    out = cv2.VideoWriter('third_eye_tracker.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1],size[0]))
    for i in range(len(images)):
        out.write(images[i])
    out.release()

if __name__ == '__main__':

    # Initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    print("loading.. yolo")    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


    cap = cv2.VideoCapture(path)

    while True:

        success, frame = cap.read()
        if not success:
            print("No video found!")
            break
        
        # frame = resize_with_padding(Image.fromarray(frame), dim)  
        obs = [0,0,0]
        warn = [" ", " ", " "]

        results = model(frame)
        box_img, left, right = draw_ROI(frame, roi)
        det = results.xyxy[0]
        if det is not None and len(det):
            x1y1x2y2 = det[:,0:4]
            xywhs = x1y1x2y2_to_xywh(x1y1x2y2)
            confs = det[:,4]
            clss = det[:,5]
            outputs = deepsort.update(xywhs, confs, clss, frame)
            counter += 1
            database, frame = draw_boxes(database, frame, outputs, confs, left, right, obs)
            images.append(frame)
            cv2.imshow("Object detection",frame)
            cv2.waitKey(1)

        else:
                
            deepsort.increment_ages()

    save_output(images, fps = 29)


