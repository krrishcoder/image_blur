import cv2 as cv
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for all origins (you can restrict this in production)
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

    # Save the uploaded file
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Process the image
    img = cv.imread(file_path)
    faces = RetinaFace.detect_faces(file_path, threshold=0.5)

    for key, value in faces.items():
        x1, y1, x2, y2 = value["facial_area"]
        roi = img[y1:y2, x1:x2]
        blurred_region = cv.GaussianBlur(roi, (15, 15), 0)
        img[y1:y2, x1:x2] = blurred_region 

    # Save the processed image
    output_path = os.path.join("uploads", "blurred_" + file.filename)
    cv.imwrite(output_path, img)

    return {"status": "Image processed", "output_url": output_path}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
