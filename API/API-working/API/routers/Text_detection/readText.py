from fastapi import APIRouter , File , UploadFile
#from dependencies.ocr import easy_ocr, pytesseract_ocr
#import cv2
from faceRecognition.FR2 import face_recognition
from currency_identification.currency_detection import currency_detection
from OCR.ocr  import *
Text_router = APIRouter()
ocr1 = OCR()

@Text_router.post("/abstract_mode/images/", tags = ["OCR"])
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()  # <-- Important!
    with open("temp.jpg", "wb") as f:
        f.write(contents)
    text = ocr1.text_detection('simple', "temp.jpg")
    return {"Text": text}

@Text_router.post("/document_mode/images/", tags = ["OCR"])
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()  # <-- Important!
    with open("temp.jpg", "wb") as f:
        f.write(contents)
    text = ocr1.text_detection('document', "temp.jpg")
    return {"Text": text}


@Text_router.post("/currency/images/", tags = ["currency"])
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()  # <-- Important!
    with open("temp.jpg", "wb") as f:
        f.write(contents)
    path = 'currency_identification/model/best.pt'
    currency = currency_detection(path)
    text = currency.main_detection(path='temp.jpg')
    return {"Text": text}

@Text_router.post("/facereco/images/", tags = ["currency"])
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()  # <-- Important!
    with open("temp.jpg", "wb") as f:
        f.write(contents)
    threshold = 1
    database = "faceRecognition/database.npy"
    faceCascade= "faceRecognition/haarcascades/haarcascade_frontalface_default.xml"
    face_recog = face_recognition(threshold= threshold, haarcascades = faceCascade, database_exist = True, database_path = database)
    test_image_path = "temp.jpg"
    text = face_recog.face_detection(source = "image", path = test_image_path)
    return {"Text": text}

