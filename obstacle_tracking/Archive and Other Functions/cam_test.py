import cv2 
from warn_json import *


def track_cam():
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

    database = {}
    cap = cv2.VideoCapture(0)
    cap.open('http://localhost:8080/iriun')

    while True:
        success, frame = cap.read()
        if not success:
            print("No video found!")
            break

        frame = cv2.cvtColor(cv2.flip(frame, 0),cv2.COLOR_BGR2RGB)
        
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
            database, frame = detect_obs(database, frame, outputs, confs, left, right, obs, warn, warn_db) #, sink, source)
            images.append(frame)
            cv2.imshow("Object detection",frame)
            cv2.waitKey(1)
        else:  
            deepsort.increment_ages()

    print(warn_count)
    save_output(images, fps = 29)

if __name__ == '__main__':
    track_cam()