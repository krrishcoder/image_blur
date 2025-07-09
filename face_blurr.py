import cv2 as cv
import numpy as np
from retinaface import RetinaFace
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64

import os
os.environ["HOME"] = "/tmp"  # Fix DeepFace PermissionError

from deepface import DeepFace


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



from fastapi.responses import Response

@app.post("/process")
async def process_post(file: UploadFile = File(...)):


    if not file:
        return JSONResponse(content={"error": "No file part"}, status_code=400)
        
    contents = await file.read()

    # Decode and process the image
    nparr = np.frombuffer(contents, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    if img is None:
        return JSONResponse(content={"error": "Could not decode image"}, status_code=400)

    # Detect faces

    faces = RetinaFace.detect_faces(img_path=img, threshold=0.5)

    for key, value in faces.items():
        x1, y1, x2, y2 = value["facial_area"]
        roi = img[y1:y2, x1:x2]
        blurred_region = cv.GaussianBlur(roi, (15, 15), 0)
        img[y1:y2, x1:x2] = blurred_region

    # Encode to JPEG
    _, buffer = cv.imencode(".jpg", img)

    # Return as image/jpeg response
    return Response(content=buffer.tobytes(), media_type="image/jpeg")
