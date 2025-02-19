from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
from fastapi.responses import FileResponse
from datetime import datetime

app = FastAPI()

# Define CSV file path
CSV_FILE = "sensor_data.csv"

# Create file with headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["time", "ax", "ay", "az", "wx", "wy", "wz", "Bx", "By", "Bz"])
    df.to_csv(CSV_FILE, index=False)

# Data model
class SensorData(BaseModel):
    times: int
    ax: float
    ay: float
    az: float
    wx: float
    wy: float
    wz: float
    Bx: float
    By: float
    Bz: float

@app.post("/sensor")
async def store_sensor_data(sensor: SensorData):
    try:
        # Convert incoming data to DataFrame
        df = pd.DataFrame([sensor.dict()])

        # Append data to CSV
        df.to_csv(CSV_FILE, mode="a", header=False, index=False)

        return {"status": "success", "message": "Data stored in CSV"}
    
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/download")
async def download_csv():
    """Download the CSV file"""
    return FileResponse(CSV_FILE, media_type="text/csv", filename="sensor_data.csv")
