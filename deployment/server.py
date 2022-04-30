# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import Response
# import os
# import uuid

# app = FastAPI()

# db = []


# @app.post("/images/")
# async def create_upload_file(file: UploadFile = File(...)):

#     file.filename = f"{uuid.uuid4()}.jpg"
#     contents = await file.read()  # <-- Important!

#     db.append(contents)

#     return {"filename": file.filename}

import os

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse

app = FastAPI()

path = r"C:\Users\Bindu\Pictures\Camera Roll"

@app.get("/")
def index():
    return {"Hello": "World"}

# @app.post("/image", responses={200: {"image": "1st image", "content" : {"image/jpeg" : {"ima1" : "flower"}}}})
# def image_endpoint():
#     file_path = os.path.join(path, r"C:\Users\Bindu\Downloads\download.jpg")
#     if os.path.exists(file_path):
#         return FileResponse(file_path, media_type="image/jpeg", filename="vector_image_for_you.jpg")
#     return {"error" : "File not found!"}

@app.post("/files/")
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}