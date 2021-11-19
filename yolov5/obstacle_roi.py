import torch
import cv2
import time
import numpy as np
import warnings
from PIL import Image, ImageOps

warnings.simplefilter('ignore')

video = r"data\road.mp4"

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


if __name__ == '__main__':

    ## Loading model ##
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    ## Setting parameters and variables##
    font = cv2.FONT_HERSHEY_SIMPLEX
    threshold = 0.1
    obstacles = ['car', 'person', 'motorcycle', 'train', 'truck']
    roi = 0.2
    success = True
    start_time  = time.time()
    images = []
    FPS = []

    ## Image capture ##
    cap = cv2.VideoCapture(video)
    while success:
        success , frame = cap.read()
        if not success:
            break

        ## Detection ##
        output = model(frame)      
        results = output.pandas().xyxy[0]
        end_time = time.time()
        duration = end_time - start_time
        fps = np.round(1/duration,1)
        start_time = end_time
        FPS.append(fps)

        ## Region of Interest ##
        height, width= frame.shape[:2]  
        left, right = roi * width, (1-roi) * width
        ROI_region = [[(int(roi * width),height),(int(roi * width),0),(int((1-roi) * width),0),(int((1-roi) * width),height)]]
        
        ## Drawing boxes ##
        for result in results.to_numpy():
            confidence = result[4]
            if confidence >= threshold:
                x1,y1,x2,y2,label = int(result[0]),int(result[1]),int(result[2]),int(result[3]),result[6]
                area = (x2-x1) * (y2-y1)
                area = int(area/100)
                box_img = frame
                color = (0,255,0)
                if x2>=left and x1<=right:
                    if area < 60:
                        color = (255,255,0)
                    else:
                        color = (0,0,255)
                if label in obstacles: #in ['car']:
                    box_img = cv2.rectangle(box_img, (x1,y1),(x2,y2),color,2)
                    box_img = cv2.putText(box_img,label,(x1-1,y1-1),font,0.5,color,1)
                    box_img = cv2.putText(box_img,str(area),(x2-1,y1-1),font,0.5,(255,0,255),1)
                    box_img = cv2.putText(box_img,f"FPS: {fps}",(32,32),font,0.5,(255,0,255),2)
            else:
                continue
        dim = (800, 600)
        box_img = cv2.rectangle(box_img, ROI_region[0][1],ROI_region[0][3],(0,0,0),1)
        final_img = np.asarray(resize_with_padding(Image.fromarray(box_img), dim))
        # cv2.imshow("Object detection",final_img)
        # cv2.waitKey(1)
        images.append(box_img)

    ## Saving Output ##
    size = images[0].shape
    out = cv2.VideoWriter('out/third_eye.mp4',cv2.VideoWriter_fourcc(*'mp4v'), int(np.mean(FPS)), (size[1],size[0]))
    for i in range(len(images)):
        out.write(images[i])
    out.release()
