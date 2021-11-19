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
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
font = cv2.FONT_HERSHEY_SIMPLEX
threshold = 0.1
cap = cv2.VideoCapture(video)

# # calculate the center of the image
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))


# center = (w / 2, h/ 2)

# angle270 = 270

# scale = 1.0

success = True
start_time  = time.time()
images = []
FPS = []
while success:

    success , frame = cap.read()
    

    if not success:
        break
    # M = cv2.getRotationMatrix2D(center, angle270, scale)  #uncomment if video feed is rotated
    # frame = cv2.warpAffine(frame, M, (w, h))
    output = model(frame)

    
    results = output.pandas().xyxy[0]
    end_time = time.time()
    duration = end_time - start_time
    fps = np.round(1/duration,1)
    start_time = end_time
    FPS.append(fps)
    
    for result in results.to_numpy():
        confidence = result[4]
        if confidence >= threshold:
            x1,y1,x2,y2,label = int(result[0]),int(result[1]),int(result[2]),int(result[3]),result[6]
            box_img = cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            box_img = cv2.putText(box_img,label,(x1-1,y1-1),font,0.5,(255,0,255),1)
            box_img = cv2.putText(box_img,f"FPS: {fps}",(32,32),font,0.5,(255,0,255),2)
            
        else:
            continue
    dim = (800, 600)
    final_img = np.asarray(resize_with_padding(Image.fromarray(box_img), dim))
    cv2.imshow("Object detection",final_img)
    cv2.waitKey(1)
    # images.append(box_img)

# size = images[0].shape
# out = cv2.VideoWriter('out/third_eye.mp4',cv2.VideoWriter_fourcc(*'mp4v'), int(np.mean(FPS)), (size[1],size[0]))
# for i in range(len(images)):
#     out.write(images[i])
# out.release()
