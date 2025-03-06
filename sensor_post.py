from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
from fastapi.responses import FileResponse

app = FastAPI()

# Define base CSV file naming
CSV_BASE_NAME = "sensor_data"
CSV_EXTENSION = ".csv"

# Function to generate a new filename with an incremented number
def get_new_filename():
    i = 1
    while os.path.exists(f"{CSV_BASE_NAME}_{i}{CSV_EXTENSION}"):
        i += 1
    return f"{CSV_BASE_NAME}_{i}{CSV_EXTENSION}"

# Data model
class SensorData(BaseModel):
    time: float  # Keep time as float
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
async def store_sensor_data(sensors: List[SensorData]):  # Accepts a list of sensor readings
    try:
        # Generate a new filename
        new_csv_file = get_new_filename()

        # Convert data to DataFrame
        df = pd.DataFrame([sensor.dict() for sensor in sensors])

        # Save the new file
        df.to_csv(new_csv_file, index=False)

        # Remove old CSV files (keep only the latest)
        for file in os.listdir():
            if file.startswith(CSV_BASE_NAME) and file != new_csv_file:
                os.remove(file)

        return {"status": "success", "message": f"Data stored in {new_csv_file}"}
    
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/download")
async def download_csv():
    """Download the latest CSV file"""
    csv_files = sorted(
        [f for f in os.listdir() if f.startswith(CSV_BASE_NAME) and f.endswith(CSV_EXTENSION)],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),  # Sort by file number
        reverse=True  # Get the latest file
    )

    if csv_files:
        latest_file = csv_files[0]  # Pick the most recent CSV file
        return FileResponse(latest_file, media_type="text/csv", filename=latest_file)

    return {"status": "error", "message": "No file available for download"}

