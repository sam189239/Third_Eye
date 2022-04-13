# from flask import Flask, Response, request
# import requests
# import os
# import json
# import uuid

# app = Flask(__name__)

# @app.route('/upload', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         file = request.files['file']
#         extension = os.path.splitext(file.filename)[1]
#         f_name = str(uuid.uuid4()) + extension
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name))
#         return json.dumps({'filename':f_name})

# if __name__ == '__main__':
# 	app.run(debug=True, host='0.0.0.0')

## pip install fastapi, python-mulipart
# pip install "uvicorn[standard]"

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import os
import uuid

app = FastAPI()

db = []


@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):

    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()  # <-- Important!

    db.append(contents)

    return {"filename": file.filename}

 