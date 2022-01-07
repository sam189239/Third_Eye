
from numpy.lib import arraysetops
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
import time

import pyttsx3 as tts


from playsound import playsound
  


## Setting parameters and variables##
save_dir =r'out\third_eye_tracker1.mp4'
path = r"..\data\VID-20211208-WA0001.mp4"
deep_sort_weights = 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'
config_deepsort="deep_sort_pytorch/configs/deep_sort.yaml"
font = cv2.FONT_HERSHEY_DUPLEX
obstacles = ['car', 'person', 'motorcycle','truck','bicycle', 'parking meter', 'cow', 'dog']
roi = 0.35
ext_roi = 0.1
success = True
threshold = 0.3
size_threshold = {'car':60, 'person':40, 'motorcycle':40,'truck':80,'bicycle':40, 'parking meter':0, 'cow':0, 'dog':0}
size_threshold_outside_roi = {'car':220, 'person':80, 'motorcycle':150,'truck':250,'bicycle':150, 'parking meter':0, 'cow':0, 'dog':0}
images = []
dim = (640, 480)
database = {}
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

traj_len = 50
warn_avg_size = 80
del_angle_threshold = 0.2
x_range = 10
z_const = 4
obs_hist = [0,0,0]
warn_count = 0

def new_obs(label):
    return {
        "c_hist":deque(maxlen=2), 
        "area_hist":deque(maxlen=warn_avg_size), 
        "angle_hist":deque(maxlen=warn_avg_size), 
        "large":False, 
        "label":label,
        "del_area":0,
        "del_angle":0,
        "warning":deque(maxlen=warn_avg_size)
    }

def find_angle(mid, height, xc, yc):
    return np.abs(math.atan((mid - xc) / ((height - yc) + 0.001)) * (180/math.pi))

def track_obs(database, id, xc, yc, frame):
    height, width= frame.shape[:2] 
    mid = int(width / 2)
    
    ## Trajectory ##
    database[id]["c_hist"].append((int(xc),int(yc)))
    c_hist = database[id]["c_hist"]
    if len(c_hist) == 2:
        lenab = math.sqrt((c_hist[0][0]-c_hist[1][0])**2+(c_hist[0][1]-c_hist[1][1])**2)
        length = 100

    ## Angle ##
    database[id]['angle_hist'].append(find_angle(mid, height, xc, yc))
    # frame = cv2.putText(frame,str(database[id]['angle_hist'][-1]),(xc,yc),font,0.5,(255,0,255),1)
    del_angle= get_del(database[id]['angle_hist'])
    database[id]['del_angle'] = del_angle
    frame = cv2.putText(frame,str(format(database[id]['del_angle'],".2f")),(xc,yc),font,0.5,(255,0,255),1)
    return database, frame

def get_del(vals):
    if(len(vals)<=1):
        return 0
    delsum = 0
    for i in range(1, len(vals)):
        delsum += vals[i] - vals[i-1]
    return delsum / len(vals)

def detect_obs(database, frame, outputs, confs, left, right, obs, warn, warn_db): #, sink, source):
    mid = int((left+right)/2)
    height, width= frame.shape[:2]

    

    if len(outputs) > 0:
        for j, (output, conf) in enumerate(zip(outputs, confs)):  
            label = names[int(output[5])]  # integer class
            if conf > threshold and (label in obstacles):                   
                bboxes = output[0:4]
                x1,y1,x2,y2 = int(bboxes[0]),int(bboxes[1]),int(bboxes[2]),int(bboxes[3])
                id = output[4]
                xc = int(x1+(x2-x1)/2) # center-x
                yc = int(y1 + (y2-y1)/2) # center-y
                area = int(((x2-x1) * (y2-y1))/100)
                color = (0,255,0)
                # height, width= frame.shape[:2]
                left_ext, right_ext = int(ext_roi*width), int((1-ext_roi) * width) # external roi
                if area>size_threshold[label] and (xc>=left_ext and xc<=right_ext):
                    ## Creating obs in database
                    if id not in database.keys():
                        database[id] = new_obs(label)
                    
                    ## Area ##
                    database[id]['area_hist'].append(area)
                    database[id]['del_area']= get_del(database[id]['area_hist'])

                    ## Obstacle ##
                    database, frame = track_obs(database, id, xc, yc, frame)

                    ## Based on del_area and del_angle ##
                    if x2>=left and x1<=right:
                        color = (0,255,255)    
                        if database[id]["del_angle"] < del_angle_threshold and database[id]['del_area'] > 0: 
                            # Within ROI, angle dec, size inc
                                database[id]["warning"].append(True)       
                        else:
                            database[id]["warning"].append(False)
                    elif (database[id]['del_angle'] < 0 and database[id]['del_area'] > 0) and area >= size_threshold_outside_roi[label]:  ## outside roi
                        # Outside ROI, angle dec, size inc, size beyond threshold
                        database[id]["warning"].append(False) # Should set to True to warn for obstacles outside ROI
                    else:
                        database[id]["warning"].append(False)

                    if np.sum(database[id]["warning"])>=(warn_avg_size/2):
                        color = (0,0,255) # Warning
                        warn_db[id] = [xc,yc]
                        if x2 <= mid and x2 >= left: ## Left (in ROI)
                            obs[0] += 1
                        elif x1 >= mid and x1 <= right: ## Right (in ROI)
                            obs[1] += 1
                        else:
                            obs[2] += 1

                    disp = f'{label} {area} {conf:.2f}'
                    cv2.rectangle(frame, (x1,y1),(x2,y2),color,2)
                    cv2.putText(frame,disp,(x1-1,y1-1),font,0.5,(255,0,255),1)

                    
                    refpt = (int(width/2),int(height))
                    dist = np.round(np.sqrt((refpt[0] - yc)**2 + (refpt[1] - xc)),1)
                    cv2.line(frame,refpt , (int(width/2),0), (0,0,234),2)
                    cv2.arrowedLine(frame, refpt,(xc,yc),(123,232,324), 1)
                            
                
    obs_current = [0,0,0]
    ## Warning ##
    for i in range(3):
        if obs[i] >= 1:
            warn[i] += "WARNING"
            obs_current[i] = 1
    cv2.putText(frame,str(obs[0]) + warn[0],(left,frame.shape[:2][0]),font,0.5,(0,0,255),2)
    cv2.putText(frame,str(obs[2]) + warn[2],(mid,frame.shape[:2][0]),font,0.5,(0,0,255),2)
    cv2.putText(frame,str(obs[1]) + warn[1],(right,frame.shape[:2][0]),font,0.5,(0,0,255),2)

    warn_color = (0,255,0)
    global obs_hist, warn_count
    if obs_hist != obs_current:
        print(obs_current)
        warn_count += 1

    if obs_current[2] > 0: # mid
        warn_color = (0,0,255)
        if obs_hist[2] != obs_current[2]: # no warning in prev frame
            playsound('warning.wav')  
    elif obs_current[0] and obs_current[1]:# left and right
        warn_color = (0,0,255)
        if (obs_hist[1] != obs_current[1]) or (obs_hist[0] != obs_current[0]): # no warning in prev frame
            playsound('warning.wav')
    elif obs_current[0] > 0:
        warn_color = (0,255,255)
        if obs_hist[0] != obs_current[0]:
            playsound('left.wav')
    elif obs_current[1] > 0: 
        warn_color = (0,255,255)
        if obs_hist[1] != obs_current[1]:
            playsound('right.wav')
    else:
        warn_color = (0,255,0)
    
    obs_hist = obs_current
    cv2.circle(frame,(int(0.95 * width), int(0.90 * height)), int(0.01 * (height + width)), warn_color, -1)

    return database, frame

def draw_ROI(frame,roi, ext_roi):
    height, width= frame.shape[:2]  
    left, right = int(roi * width), int((1-roi) * width)
    mid = int(width / 2)
    ROI_region = [[(left,height),(left,0),(right,0),(right,height)]]
    ROI_region2 = [[(left,height),(left,0),(mid,0),(mid,height)]]
    cv2.rectangle(frame, ROI_region[0][1],ROI_region[0][3],(0,0,0),1)
    cv2.rectangle(frame, ROI_region2[0][1],ROI_region2[0][3],(0,0,0),1)
    cv2.rectangle(frame, (int(ext_roi*width),0),(int((1-ext_roi) * width), height),(0,0,0),1)
    return frame, left, right

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
    out = cv2.VideoWriter(save_dir,cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1],size[0]))
    for i in range(len(images)):
        out.write(images[i])
    out.release()

def track():
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


    database = {}
    cap = cv2.VideoCapture(path)

    while True:
        success, frame = cap.read()
        if not success:
            print("No video found!")
            break
        
        obs = [0,0,0]
        warn = [" ", " ", " "]
        warn_db = {}
        frame, left, right = draw_ROI(frame, roi, ext_roi)
        # frame = resize_with_padding(Image.fromarray(frame), dim)
        results = model(frame)
        
        det = results.xyxy[0]
        if det is not None and len(det):
            x1y1x2y2 = det[:,0:4]
            xywhs = x1y1x2y2_to_xywh(x1y1x2y2)
            confs = det[:,4]
            clss = det[:,5]
            outputs = deepsort.update(xywhs, confs, clss, frame)
            database, frame = detect_obs(database, frame, outputs, confs, left, right, obs, warn, warn_db) #, sink, source)
            images.append(frame)
            cv2.imshow("Object detection",frame)
            cv2.waitKey(1)

        else:
                
            deepsort.increment_ages()
    print(warn_count)
    save_output(images, fps = 29)


if __name__ == '__main__':
    track()


