from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
from fastapi.responses import FileResponse

app = FastAPI()

# Define the base file name
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
        # Generate a new filename
        new_csv_file = get_new_filename()

        # Convert incoming data to DataFrame
        df = pd.DataFrame([sensor.dict()])

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
    latest_file = get_new_filename()  # Get the latest file name
    latest_file_number = int(latest_file.split("_")[-1].split(".")[0]) - 1
    latest_file = f"{CSV_BASE_NAME}_{latest_file_number}{CSV_EXTENSION}"

    if os.path.exists(latest_file):
        return FileResponse(latest_file, media_type="text/csv", filename=latest_file)
    return {"status": "error", "message": "No file available for download"}
