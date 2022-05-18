import cv2
import pytesseract
import easyocr


    
def easy_ocr(picture):
    text = []
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(picture)
    for i in result:
        if i[2] > 0.25: # i[2] confidence level
            text.append(i[1]) # i[1] word/letter
    text = " ".join(text)
    return text 
    
def pytesseract_ocr(picture):

    
    pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

    text = pytesseract.image_to_data(picture, output_type='data.frame')
    text = " ".join(list(text[text['conf']>40]['text']))
    return text


            
