import cv2 as cv
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process")
async def process_post(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(content={"error": "No file part"}, status_code=400)

    # Read uploaded image into memory as bytes
    contents = await file.read()
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

    # Encode image to memory as JPEG and then base64
    _, buffer = cv.imencode('.jpg', img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return {
        "status": "Image processed",
        "image_base64": f"data:image/jpeg;base64,{encoded_image}"
    }
