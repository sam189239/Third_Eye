import cv2
import pyttsx3
import pytesseract
import easyocr

from gtts import gTTS
from playsound import playsound 
import os


class OCR:

    global clicked
    clicked = False

    def __init__(self):
        self.engine = pyttsx3.init()

    def speaker(self, text):
        # self.engine.setProperty('rate', 178)
        # self.engine.say(text)
        # self.engine.runAndWait()

        tts = gTTS(text, lang='en') # 'hi' for hindi, de for german, en for english
        filename = "abc.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)

    def blur_detection(self, image):
        blur_val = cv2.Laplacian(image, cv2.CV_64F).var()
        print(blur_val)
        if blur_val < 17:
            return True
        return False

    def easy_ocr(self, picture):
        text =[]
        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(picture)
        for i in result:
            if i[2] > 0.25: # i[2] confidence level
                text.append(i[1]) # i[1] word/letter
        text = " ".join(text)
        return text

    def pytesseract_ocr(self, picture):
        # pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"
        text = pytesseract.image_to_data(picture, output_type='data.frame')
        text = " ".join(list(text[text['conf']>40]['text']))
        return text

    def mode_selection(self):
        self.speaker("What do you want to read? select the mode: document or simple")
        mode = input("Simple or Document?: ")
        mode = mode.lower()
    
        while mode not in ['document', 'simple']:
            self.speaker("please enter a valid mode of reading between simple and document") 
            mode = input("Simple or Document?: ")
            mode = mode.lower()
       
        self.speaker(mode + " mode selected")
        return mode

    def mouse_click(self, event, x, y, flags, param):
        global clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked = True

    def text_detection(self, mode, path):
        img = cv2.imread(path)
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
        if mode == 'document' and not self.blur_detection(frame):
           #print('document mode selected')
           text1='>>>Doc mode: '
           text2 = self.pytesseract_ocr(frame)
           text = text1 + text2
           return text
           #self.speaker(text + ". text detection complete")
                                           
        elif mode == 'simple' and not self.blur_detection(frame):
           #print('simple mode selected')
           text1= '>>>Simple Mode: '
           text2 = self.easy_ocr(frame)
           text = text1 + text2
           return text
           #print(text)
           #self.speaker(text + ". text detection completed")
                                                  
        else:
           text = 'no detection'
           return text
           #self.speaker("blurry image... please try again.")        
