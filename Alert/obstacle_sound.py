import torch
import cv2
import time
import numpy as np
import warnings
from PIL import Image, ImageOps
import time
import math
from openal.audio import SoundSink, SoundSource
from openal.loaders import load_wav_file
from openal.al import *
from openal.alc import *

warnings.simplefilter('ignore')

video = r"..\data\VID-20211208-WA0001.mp4"

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

x_range = 10
def translate_xy(x, y, h, w):
    y_range = int(h/w * x_range)
    x = int((int(x / w) * x_range) - (x_range/2))
    y = int((int(y / h) * y_range) - (y_range/2))
    return x, y

# source = []


def warn_sound(x, y, h, w, sink, source):
    x,y = translate_xy(x, y, h, w)    
    source.position = [x, y, z_const]      
    sink.update()


# def warn_sound2(warn_db, h, w):
#     # for s in source:
#     #     # sink.stop(SoundSource(s))
#     # source.clear()
    
#     for a in warn_db.keys():
#         warn_db[a][0],warn_db[a][1] = translate_xy(warn_db[a][0],warn_db[a][1], h, w)    
#         source.append((a,SoundSource(position=[warn_db[a][0],warn_db[a][1],z_const])))        
    
#     for s in source:
#         s[1].looping = True
#         s[1].queue(data)
#         print(type(s[1]))
#         sink.play(SoundSource(s[1]))
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

    sink = SoundSink()
    sink.activate()
    source1 = SoundSource(position=[0, 0, 0])
    source2 = SoundSource(position=[0, 0, 4])
    source1.looping = True
    source2.looping = True
    data = load_wav_file("bounce.wav")
    source1.queue(data)
    source2.queue(data)
    # sink.play(source1)
    sink.play(source2)
    t=0
    ## Image capture ##
    cap = cv2.VideoCapture(video)
    while success:
        sink.update()
        print("playing at %r" % source1.position)
        time.sleep(0.1)
        t += 5
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
        ROI_region2 = [[(int(roi * width),height),(int(roi * width),0),(int((0.5) * width),0),(int((0.5) * width),height)]]
        
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
        # box_img = cv2.rectangle(box_img, ROI_region2[0][1],ROI_region2[0][3],(0,0,0),1)
        final_img = np.asarray(resize_with_padding(Image.fromarray(box_img), dim))
        cv2.imshow("Object detection",final_img)
        cv2.waitKey(1)
        images.append(final_img)
        # warn_sound(output[0],output[1], height, width, sink, source)

    ## Saving Output ##
    size = images[0].shape
    out = cv2.VideoWriter('out/third_eye.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 29, (size[1],size[0]))
    for i in range(len(images)):
        out.write(images[i])
    out.release()