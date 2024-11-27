from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from deepface import DeepFace

app = FastAPI()

class ImageData(BaseModel):
    image: str 
    
@app.get("/")
def read_root():
    return {"message" : "This is face emotion detection API for embeded system project"}

@app.post("/detect")
async def detect_emotion(data: ImageData):
    try:
        image_data = data.image
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        result = DeepFace.analyze(img, actions=['emotion'])
        emotion = result[0]['dominant_emotion']
        return {"emotion": emotion}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
