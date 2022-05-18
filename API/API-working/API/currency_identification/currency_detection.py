import cv2
import torch
import pyttsx3
import warnings
warnings.filterwarnings("ignore")

class currency_detection:

    def __init__(self,path):
        
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
        self.engine = pyttsx3.init()

    def most_frequent(self,List):
        '''
        finding most frequent element in a nested list
        input: list
        output: most frequent element
        '''
        counter = 0
        num = List[0]
        
        for i in List:
            curr_frequency = List.count(i)
            if(curr_frequency> counter):
                counter = curr_frequency
                num = i
    
        return num
    

    def detect_currency(self,frame):
        '''
        detect currency notes from captured frames
        input: frame
        '''
        names=[]
        result = self.model(frame)
        print(result.pandas().xyxy[0])
        labels, cord_thres = result.xyxyn[0][:, -1].cpu().numpy(), result.xyxyn[0][:, :-1].cpu().numpy()
        print(labels)
        print(" ".join(self.model.names))
        for i in range(labels.size):
            row = cord_thres[i]
            if row[4] >= 0.5:
               names.append((self.model.names[int(labels[i])]))
                    #else:
                    #    names.append(0)
               
        str1= " "
        text = 'currency is : ' + str1.join(names)
        return text
        #self.text2speech(names)

    def mouse_click(self,event,x,y,flags,param):
        global clicked
        if event == cv2.EVENT_LBUTTONDOWN:  
            clicked = True
    

    def main_detection(self,path):
       img = cv2.imread(path)
       #cv2.imshow("Currency", img)
       #cv2.waitKey(0)
       #cv2.destroyAllWindows()
       #print(img)
       text = self.detect_currency(img)
       return text
