import cv2
import torch
import numpy as np
import warnings
import pyttsx3

warnings.filterwarnings("ignore")

class everydayobjectdetection:

    def __init__(self, object_threshold, show_video=True):
        self.object_threshold = object_threshold
        self.show_video = show_video
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
        self.engine = pyttsx3.init()
        self.prevstate = " "

    def text2speech(self,text):
        self.engine.say(text)
        self.engine.runAndWait()

    def find_crosshair(self,bboxes,frame_center):
        '''
        To check whether center of frame is inside the object bounding box area.
        '''
        in_center = []
        for i in range(len(bboxes)):
            if ( (bboxes[i][1] < frame_center[0] < bboxes[i][3]) and ( bboxes[i][2] < frame_center[1] < bboxes[i][4])) :
                in_center.append(bboxes[i])
            else:
                in_center.append(0)
        return in_center

    def load_class_threshold(self):
        '''
        load pre-registered object areas 
        output : class thresholds
        '''
        temp = []
        stored_file = np.load(open(self.object_threshold, 'rb'),allow_pickle=True)   #### load different object areas
        for i in range(len(stored_file)):
            temp.append([stored_file[i][0],stored_file[i][1]])
            
        class_thresholds = {temp[i][0]: temp[i][1] for i in range(len(temp))}
        return class_thresholds

    def trigger_ROI(self,bboxes): 
        '''
        This method decide compare detected object area and pre-regisitered
        object area. if the detected object area is less than pre-registered 
        '''
        class_thresholds = self.load_class_threshold()

        for i in range(len(bboxes)):
            class_id = bboxes[i][0] 
            objw = bboxes[i][3]-bboxes[i][1]
            objh = bboxes[i][4]-bboxes[i][2]
            obj_area = objw * objh
            if class_id in class_thresholds:
                if obj_area < class_thresholds.get(class_id):
                    bboxes[i] = 0     # replace the object bboxes with 0
            else:
                bboxes[i] = 0
        return bboxes

    def calculate_bbox(self,labels,cord_thres,image):
        '''
        To get detected object bounding boxes as a list
        '''
        x_shape, y_shape = image.shape[1], image.shape[0]
        bboxes = []
        for i in range(len(labels)):
            x1,y1,x2,y2 = int(cord_thres[i][0]*x_shape), int(cord_thres[i][1]*y_shape), int(cord_thres[i][2]*x_shape), int(cord_thres[i][3]*y_shape)
            bboxes.append([int(labels[i]),x1,y1,x2,y2])
        return bboxes
                    
    def ROI_detect(self,image,frame_center):
        
        results = self.model(image)
        labels, cord_thres = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        if labels.size == 0:
            object_name = 'No object!'
    
        else:
            bboxes = self.calculate_bbox(labels, cord_thres, image)
            bboxes = self.trigger_ROI(bboxes)
            bboxes = [i for i in bboxes if i != 0] # remove all 0 in bboxes
            if len(bboxes) != 0:
                bboxes = self.find_crosshair(bboxes,frame_center)
                bboxes = [i for i in bboxes if i != 0]
                if len(bboxes) != 0:
                    
                    object_name = self.model.names[bboxes[0][0]] + ' is detected'
                    cv2.rectangle(image,(bboxes[0][1], bboxes[0][2]), (bboxes[0][3], bboxes[0][4]),(255, 0, 0), 2)
                    cv2.putText(image,self.model.names[bboxes[0][0]],(bboxes[0][1],bboxes[0][2] ),cv2.FONT_HERSHEY_SIMPLEX,0.9, (255, 255, 0), 2)
                else:
                    object_name = 'No object!'
            else:
                object_name = 'No object!'
                
        return image,object_name

    def detect_object(self,videopath,res):
        cam_capture = cv2.VideoCapture(videopath)
        timer = 0
        if cam_capture.isOpened():
            # calculate frame width, height and center point
            width = res[0]
            height = res[1]
            x = int(width/2)
            y = int(height/2)
            frame_center = (x,y)
            while True:
                try:
                    is_success, image_frame = cam_capture.read()        
                except cv2.error:
                    continue
                if not is_success:
                    break
                image_frame = cv2.resize(image_frame, res, interpolation = cv2.INTER_AREA)
                image_frame,result = self.ROI_detect(image_frame,frame_center)
                print(result)
                
                if self.prevstate == result:
                    if timer > 20:  # 20 seconds delay
                        self.text2speech(result)
                        timer = 0    
                    else:
                        timer += 1
                else:
                    self.text2speech(result)
                self.prevstate = result

                if self.show_video:
                    cv2.circle(image_frame, frame_center, radius=5, color=(0, 0, 255), thickness=-1)
                    cv2.imshow("Focus Mode", image_frame)
                    if cv2.waitKey(1) == 13:  ## enter key to stop 
                        break
                
        cam_capture.release()
        cv2.destroyAllWindows()

class find_object_area:

    def __init__(self):
        self.threshold = []
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        
    def calculate_bbox(self,labels,cord_thres,image):
        x_shape, y_shape = image.shape[1], image.shape[0]
        n = len(labels)
        confidence = []
        bboxes = []
        for i in range(n):
            confidence.append(cord_thres[i][4])
        max_con = confidence.index(max(confidence))
        x1,y1,x2,y2 = int(cord_thres[max_con][0]*x_shape), int(cord_thres[max_con][1]*y_shape), int(cord_thres[max_con][2]*x_shape), int(cord_thres[max_con][3]*y_shape)
        
        bboxes.append([int(labels[max_con]),x1,y1,x2,y2])
        return bboxes
    
    def ROI_detect(self,image):
        
        results = self.model(image)
        detected = np.squeeze(results.render())
        labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        if labels.size != 0:
            
            bboxes = self.calculate_bbox(labels, cord_thres, detected)
            name = self.model.names[bboxes[0][0]]
            objw = bboxes[0][3]-bboxes[0][1]
            objh = bboxes[0][4]-bboxes[0][2]
            obj_area = objw * objh
            
            return name,bboxes[0][0],obj_area

    def object_area(self,videopath,res):

        def mouse_click(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                name,label,area = self.ROI_detect(image_frame)
                self.threshold.append([label,area])
                print(name)
                return self.threshold  

        cam_capture = cv2.VideoCapture(videopath)
        if cam_capture.isOpened():
            while True:
                try:
                    is_success, image_frame = cam_capture.read()        
                except cv2.error:
                    continue
                if not is_success:
                    break
                image_frame = cv2.resize(image_frame, res, interpolation = cv2.INTER_AREA)
                cv2.imshow("image", image_frame)
                cv2.setMouseCallback('image', mouse_click)
                np.array(self.threshold).dump(open('objects_areas.npy', 'wb'))
                if cv2.waitKey(1) == 13:  ## enter key to stop 
                    break 
                        
        cam_capture.release()
        cv2.destroyAllWindows()


            
                
                    