from yolov5.utils.downloads import attempt_download
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from functions import *
warnings.filterwarnings("ignore")
import subprocess

## Setting parameters and variables##

input_dir = r"..\data\VID-20220108-WA0006.mp4"
output_dir =r'out\third_eye_tracker.mp4'

with open("obs_state.json", "w") as outfile:
    outfile.write(json.dumps({"state": [0, 0, 0, False]}))

roi = 0.35
ext_roi = 0.1
# success = True
conf_threshold = 0.3
size_threshold = {'car':60, 'person':40, 'motorcycle':40,'truck':80,'bicycle':40, 'parking meter':0, 'cow':0, 'dog':0}
size_threshold_outside_roi = {'car':220, 'person':80, 'motorcycle':150,'truck':250,'bicycle':150, 'parking meter':0, 'cow':0, 'dog':0}

warn_avg_size = 30
del_angle_threshold = 0.2
del_area_threshold = 0.35
crowd_threshold = 2


def alert_func():
    subprocess.Popen("python alert_3dsound.py", shell=True)


def detect_obs(database, frame, outputs, confs, left, right, obs, warn, warn_db):
    
    '''
    Performs the main algorithm to detect whether an object is a potential obstacle to the user.
    Confidence threshold is applied.
    Initially, objects within the external ROI are taken into consideration along with application of size threshold.
    Obstacle is entered into the obstacle database.
    Area and angle calculations are made to check if they are increasing or decreasing.
    Increasing area and decreasing angle is an indicator for 
    A warning deque is maintained for each obstacle where TRUE or FALSE is appended for each frame.
    Alert and Color change are invoked when more than half of the warning deque is TRUE.
    This is updated in the obs and obs_current arrays. obs -> count of obsatacles in left, right and mid. obs_current -> boolean array indicating obstacle presence in left, right and mid
    Previous obstacle state is stored in obs_hist and any variation from this is updated byu sending current state to a JSON file obs_state.json which is used by alert_3dsound.py to give audio alerts.
    An additional overall warning circle is also added to the image output

    '''

    count_of_obstacles = 0

    ## Dimensions of the frame ##
    mid = int((left+right)/2) 
    height, width= frame.shape[:2]

    ## Iterating through the outputs of the YOLO model ##
    if len(outputs) > 0:
        for j, (output, conf) in enumerate(zip(outputs, confs)):  
            label = names[int(output[5])]  
            if conf > conf_threshold and (label in obstacles):    # Checking confidence threshold              
                bboxes = output[0:4]    # Bounding box
                x1,y1,x2,y2 = int(bboxes[0]),int(bboxes[1]),int(bboxes[2]),int(bboxes[3])
                id = output[4]    # Deep sort ID
                xc = int(x1+(x2-x1)/2)    # x co-ordinate of the center of bounding box
                yc = int(y1 + (y2-y1)/2)    # y co-ordinate of the center of bounding box
                area = int(((x2-x1) * (y2-y1))/100)    # Area of the bounding box scaled by 100
                color = (0,255,0)
                left_ext, right_ext = int(ext_roi*width), int((1-ext_roi) * width) # external region of interest
                
                if area>size_threshold[label] and (xc>=left_ext and xc<=right_ext):     # Object within external ROI and area greater than threshold
                    
                    # Creating obs in database
                    if id not in database.keys():
                        database[id] = new_obs(label, warn_avg_size)
                    
                    ## Area ##
                    database[id]['area_hist'].append(area)
                    database[id]['del_area']= get_del(database[id]['area_hist'])

                    ## Angle ##
                    database, frame = update_angle(database, id, xc, yc, frame)

                    ## Based on del_area and del_angle ##
                    if x2>=left and x1<=right:    # Object inside ROI
                        count_of_obstacles += 1
                        color = (0,255,255)    
                        if database[id]["del_angle"] < del_angle_threshold and database[id]['del_area'] > del_area_threshold: 
                            # Within ROI, angle decreasing, size increasing
                                database[id]["warning"].append(True)    # Appending warning to deque
                        else:
                            database[id]["warning"].append(False)
                    elif (database[id]['del_angle'] < 0 and database[id]['del_area'] > 0) and area >= size_threshold_outside_roi[label]:    # Outside roi but area greater than second area threshold (too big / near)
                        # Outside ROI, angle dec, size inc, size beyond threshold
                        database[id]["warning"].append(False)    # Should set to True to warn for obstacles outside ROI
                    else:
                        database[id]["warning"].append(False)

                    if np.sum(database[id]["warning"])>=(warn_avg_size/2):
                        color = (0,0,255)    # Warning
                        warn_db[id] = [xc,yc]
                        if x2 <= mid and x2 >= left:    # Left (in ROI)
                            obs[0] += 1
                        elif x1 >= mid and x1 <= right:    # Right (in ROI)
                            obs[1] += 1
                        else:    # Middle
                            obs[2] += 1
                    
                    # Printing bounding boxes
                    temp = np.round(database[id]['del_area'], 2)
                    disp = f'{id} {temp} {conf:.2f}'
                    cv2.rectangle(frame, (x1,y1),(x2,y2),color,2)
                    cv2.putText(frame,disp,(x1-1,y1-1),font,0.5,(255,0,255),1)

                    # Line from centre of obstalce to user reference point (bottom mid)
                    refpt = (int(width/2),int(height))
                    cv2.line(frame,refpt , (int(width/2),0), (0,0,234),2)
                    cv2.arrowedLine(frame, refpt,(xc,yc),(123,232,324), 1)  

    obs_current = [0,0,0]

    ## Warning ##
    # Print warning in img output
    for i in range(3):
        if obs[i] >= 1:
            warn[i] += "WARNING"
            obs_current[i] = 1
    cv2.putText(frame,str(obs[0]) + warn[0],(left,frame.shape[:2][0]),font,0.5,(0,0,255),2)
    cv2.putText(frame,str(obs[2]) + warn[2],(mid,frame.shape[:2][0]),font,0.5,(0,0,255),2)
    cv2.putText(frame,str(obs[1]) + warn[1],(right,frame.shape[:2][0]),font,0.5,(0,0,255),2)

    warn_color = (0,255,0)
    
    ## Sending current obstacle state to json ##
    global obs_hist, warn_count
    obs_current.append(count_of_obstacles>=crowd_threshold)

    if obs_hist != obs_current:
        send_state(obs_current)
        print(obs_current)
        warn_count += 1

    ## Warning circle in image output ##
    if obs_current[2] > 0: # mid
        warn_color = (0,0,255)
        if obs_hist[2] != obs_current[2]: # new warning if no warning in prev frame
            pass
    elif obs_current[0] > 0 or obs_current[1] > 0: # left and right
        warn_color = (0,255,255)
    else:
        warn_color = (0,255,0)
    
    obs_hist = obs_current
    cv2.circle(frame,(int(0.95 * width), int(0.90 * height)), int(0.01 * (height + width)), warn_color, -1)

    return database, frame


def track(input_dir, output_dir = output_dir):

    '''
    Initializes the Deepsort and YOLO model to obtain bounding boxes and object id.
    Output is passed to detect_obs to process obstacles in the frame
    Final output is saved

    '''

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

    alert_func() 

    database = {}
    cap = cv2.VideoCapture(input_dir)

    while True:
        success, frame = cap.read()
        if not success:
            print("No video found!")
            break
        
        obs = [0,0,0]
        warn = [" ", " ", " "]
        warn_db = {}
        frame, left, right = draw_ROI(frame, roi, ext_roi)
        results = model(frame)
        det = results.xyxy[0]

        if det is not None and len(det):
            x1y1x2y2 = det[:,0:4]
            xywhs = x1y1x2y2_to_xywh(x1y1x2y2)
            confs = det[:,4]
            clss = det[:,5]
            outputs = deepsort.update(xywhs, confs, clss, frame)
            database, frame = detect_obs(database, frame, outputs, confs, left, right, obs, warn, warn_db)
            images.append(frame)
            cv2.imshow("Object detection",frame)
            cv2.waitKey(1)
        else:  
            deepsort.increment_ages()
    cap.release()
    cv2.destroyAllWindows()
    print(warn_count)
    save_output(images, 29, output_dir)
    with open("obs_state.json", "w") as outfile:
        outfile.write(json.dumps({"state": [0, 0, 0, False]}))


if __name__ == '__main__':
    track(input_dir, output_dir)