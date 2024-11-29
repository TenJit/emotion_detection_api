from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from deepface import DeepFace
from pymongo import MongoClient
from pymongo.mongo_client import MongoClient
from bson.objectid import ObjectId
from datetime import datetime , timedelta
import os
from dotenv import load_dotenv

app = FastAPI()

try:
    load_dotenv()
    mongodb_uri = os.getenv("MONGODB_URI")    
    print(f"Connecting to MongoDB using URI: {mongodb_uri}")
    client = MongoClient(mongodb_uri)
    client.admin.command('ping')  # Test the connection
    print("Successfully connected to MongoDB! & connection strings = ",mongodb_uri)
except Exception as e:
    print(f"Error occurred while connecting to MongoDB: {e}")
    raise

db = client["emotion_detection"]  # Database name
emotions_collection = db["emotions"]  # Collection name
water_collection = db["water"]
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
        inserted_id = emotions_collection.insert_one(emotion_data).inserted_id
        
        return {"emotion": emotion , "object_id": str(inserted_id)}

    except Exception as e:
        if str(e) == "Face could not be detected in numpy array.Please confirm that the picture is a face photo or consider to set enforce_detection param to False.":
            return {"emotion" : "No face detected"}
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/water")
async def get_water_data():
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        water_data = water_collection.find_one({"date": current_date})
        
        if not water_data:
            water_data = {
                "date": current_date,
                "water_time": []
            }
            water_collection.insert_one(water_data)
            water_data = water_collection.find_one({"date": current_date})
        
        count_happy_emotion = emotions_collection.count_documents({
            "emotion": "happy",
            "date_time": {
                "$gte": datetime.strptime(current_date, "%Y-%m-%d"), 
                "$lt": datetime.strptime(current_date, "%Y-%m-%d") + timedelta(days=1)
            }
        })
        
        if len(water_data["water_time"]) < count_happy_emotion:
            if len(water_data["water_time"]) >= 2:
                return {
                    "result": False,
                    "water_time": water_data["water_time"]
                }
            else:   
                water_collection.update_one(
                    {"_id": water_data["_id"]},
                    {
                        "$push": {"water_time": {"time": current_time}}
                    }
                )
                water_data = water_collection.find_one({"date": current_date})
                return{
                    "result": True,
                    "water_time": water_data["water_time"]
                }
        else:
            return {
                "result": False,
                "water_time": water_data["water_time"]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))