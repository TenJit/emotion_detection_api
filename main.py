from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from deepface import DeepFace
from pymongo import MongoClient
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson.objectid import ObjectId
from datetime import datetime
import os
from dotenv import load_dotenv

app = FastAPI()

try:
    mongodb_uri = os.getenv("MONGODB_URI")    
    print(f"Connecting to MongoDB using URI: {mongodb_uri}")
    client = MongoClient(mongodb_uri)
    client.admin.command('ping')  # Test the connection
    print("Successfully connected to MongoDB! & connection strings = ",mongodb_uri)
except Exception as e:
    print(f"Error occurred while connecting to MongoDB: {e}")
    raise

db = client["emotion_detection"]  # Database name
collection = db["emotions"]  # Collection name
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
        
        emotion_data = {
            "_id": ObjectId(),
            "emotion": emotion,
            "image_data": image_data,
            "date_time" : datetime.now()
        }
        inserted_id = collection.insert_one(emotion_data).inserted_id
        
        return {"emotion": emotion , "object_id": str(inserted_id)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
