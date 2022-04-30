from fastapi import APIRouter , File , UploadFile
from dependencies.ocr import easy_ocr, pytesseract_ocr
import cv2

Text_router = APIRouter()



@Text_router.post("/abstract_mode/images/", tags = ["OCR"])
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()  # <-- Important!
    with open("temp.jpg", "wb") as f:
        f.write(contents)
    image = cv2.imread("temp.jpg")
    
    
    text = easy_ocr(image)

    return {"Text": text}

@Text_router.post("/document_mode/images/", tags = ["OCR"])
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()  # <-- Important!
    with open("temp.jpg", "wb") as f:
        f.write(contents)
    image = cv2.imread("temp.jpg")
    
    
    text = pytesseract_ocr(image)

    return {"Text": text}