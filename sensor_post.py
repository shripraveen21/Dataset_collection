from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
from fastapi.responses import FileResponse
import joblib
import numpy as np
from scipy.stats import zscore, skew, kurtosis
from scipy.fft import fft
import requests

app = FastAPI()

# File and model paths
CSV_BASE_NAME = "sensor_data"
CSV_EXTENSION = ".csv"
FEATURES_FILE = "feature_extracted.csv"
MODEL_FILE = "model.pkl"
MODEL_URL = "https://github.com/shripraveen21/Dataset_collection/raw/main/model.pkl"

# Function to download the model if not present
def download_model():
    """Downloads the model.pkl file from GitHub if it's missing."""
    if not os.path.exists(MODEL_FILE):
        print("🔍 Model file not found. Downloading...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_FILE, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            print("✅ Model downloaded successfully!")
        else:
            raise FileNotFoundError("❌ Failed to download model.pkl from GitHub.")

# Ensure model exists before proceeding
download_model()

# Load the trained ML model
model = joblib.load(MODEL_FILE)
print("✅ Model loaded successfully!")

# Function to generate a new CSV filename
def get_new_filename():
    i = 1
    while os.path.exists(f"{CSV_BASE_NAME}_{i}{CSV_EXTENSION}"):
        i += 1
    return f"{CSV_BASE_NAME}_{i}{CSV_EXTENSION}"

# Data model for incoming sensor data
class SensorData(BaseModel):
    time: float  
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
async def store_sensor_data(sensors: List[SensorData]):
    """Stores incoming sensor data as a CSV file"""
    try:
        new_csv_file = get_new_filename()
        df = pd.DataFrame([sensor.dict() for sensor in sensors])
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
    """Downloads the latest stored CSV file"""
    csv_files = sorted(
        [f for f in os.listdir() if f.startswith(CSV_BASE_NAME) and f.endswith(CSV_EXTENSION)],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
        reverse=True  
    )

    if csv_files:
        latest_file = csv_files[0]
        return FileResponse(latest_file, media_type="text/csv", filename=latest_file)

    return {"status": "error", "message": "No file available for download"}

### ------------------ Z-SCORE NORMALIZATION ------------------
def normalize_data(data):
    """Applies Z-score normalization to sensor columns"""
    sensor_columns = ['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz']
    data[sensor_columns] = data[sensor_columns].apply(zscore)
    return data

### ------------------ FEATURE EXTRACTION ------------------
# Jerk Calculation Function
def compute_jerk(segment, axis):
    """Computes the Jerk Magnitude (JM) for a given axis."""
    diff = np.diff(segment[axis].to_numpy(), n=1)
    jerk = np.sqrt(diff ** 2)
    return np.mean(jerk) if len(jerk) > 0 else 0

# Feature extraction function
def extract_features(segment, axis):
    """Extracts statistical, frequency, and novel features from a single axis segment."""

    # Basic Statistical Features
    min_val = segment[axis].min()
    max_val = segment[axis].max()
    mean_val = segment[axis].mean()
    skewness = skew(segment[axis].dropna())
    kurtosis_val = kurtosis(segment[axis].dropna())
    
    # Jerk Magnitude (JM)
    jerk_mag = compute_jerk(segment, axis)
    
    # Autocorrelation Features (lags 1-11)
    autocorr_vals = [segment[axis].autocorr(lag=i) for i in range(1, 12)]
    
    # Frequency Domain Features (Top 5 frequencies and their amplitudes)
    segment_data = segment[axis].fillna(0).to_numpy()
    freq_data = np.abs(fft(segment_data))
    sorted_indices = np.argsort(freq_data)[::-1]  # Sort frequencies by magnitude, descending
    top_freqs = sorted_indices[:5]
    top_amplitudes = freq_data[top_freqs]
    
    # Frequency-Domain Peak Ratio (FDPR)
    low_freq_energy = np.sum(freq_data[:3])  # Sum of energy in 1-3 Hz range
    total_energy = np.sum(freq_data)
    fdpr = (low_freq_energy / total_energy) if total_energy > 0 else 0

    features = {
        f'{axis}_min': min_val,
        f'{axis}_max': max_val,
        f'{axis}_mean': mean_val,
        f'{axis}_skewness': skewness,
        f'{axis}_kurtosis': kurtosis_val,
        f'{axis}_jerk_magnitude': jerk_mag,
        f'{axis}_fdpr': fdpr,
    }

    # Add autocorrelation features
    for i, autocorr_val in enumerate(autocorr_vals, start=1):
        features[f'{axis}_autocorr_lag_{i}'] = autocorr_val
    
    # Add frequency and amplitude features
    for i, (freq, amp) in enumerate(zip(top_freqs, top_amplitudes), start=1):
        features[f'{axis}_freq_{i}'] = freq
        features[f'{axis}_amplitude_{i}'] = amp
    
    return pd.Series(features)

def extract_features_sliding_window(data, window_size=60, step_size=30):
    """Extracts features using a sliding window approach (without Participant_ID/Trial_Number logic)."""
    features = []
    sensor_columns = ['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz']
    
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[start:start + window_size]
        feature_row = {}

        # Extract features for each sensor axis
        for axis in sensor_columns:
            axis_features = extract_features(window, axis)
            feature_row.update(axis_features)
        
        features.append(feature_row)

    return pd.DataFrame(features)

# @app.get("/predict")
# async def run_ml_prediction():
#     """Runs ML prediction on the latest stored CSV file and returns output"""
#     csv_files = sorted(
#         [f for f in os.listdir() if f.startswith(CSV_BASE_NAME) and f.endswith(CSV_EXTENSION)],
#         key=lambda x: int(x.split("_")[-1].split(".")[0]),
#         reverse=True  
#     )

#     if not csv_files:
#         return {"status": "error", "message": "No CSV file found for prediction"}

#     latest_file = csv_files[0]
#     df = pd.read_csv(latest_file)

#     # Step 1: Apply Z-score Normalization
#     df = normalize_data(df)

#     # Step 2: Extract Features
#     feature_data = extract_features_sliding_window(df)

#     # Step 3: Save Features to CSV
#     feature_data.to_csv(FEATURES_FILE, index=False)

#     # Step 4: Perform Prediction
#     predictions = model.predict(feature_data)

#     return {"status": "success", "predictions": predictions.tolist()}
@app.get("/predict")
async def run_ml_prediction():
    """Runs ML prediction on the latest stored CSV file and returns output"""

    # Step 1: Find the latest CSV file
    csv_files = sorted(
        [f for f in os.listdir() if f.startswith(CSV_BASE_NAME) and f.endswith(CSV_EXTENSION)],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
        reverse=True  
    )

    if not csv_files:
        return {"status": "error", "message": "No CSV file found for prediction"}

    latest_file = csv_files[0]
    print(f"🔍 Using file: {latest_file}")

    # Step 2: Load CSV
    try:
        df = pd.read_csv(latest_file)
        print(f"📊 Loaded Data Shape: {df.shape}")
        print(df.head())

        if df.empty:
            return {"status": "error", "message": "CSV file is empty"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to read CSV: {e}"}

    # Step 3: Handle Missing Values (NaN)
    try:
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"⚠️ Warning: Found {missing_values} missing values in data.")
            df = df.fillna(df.mean())  # Impute missing values with column mean
            print("✅ Missing values handled using column mean.")
    except Exception as e:
        return {"status": "error", "message": f"NaN handling failed: {e}"}

    # Step 4: Apply Z-score Normalization
    try:
        df = normalize_data(df)
        print("✅ Normalization complete")
    except Exception as e:
        return {"status": "error", "message": f"Normalization failed: {e}"}

    # Step 5: Extract Features
    try:
        feature_data = extract_features_sliding_window(df)
        print(f"🧩 Extracted Feature Shape: {feature_data.shape}")
        print(feature_data.head())

        if feature_data.empty:
            return {"status": "error", "message": "Feature extraction returned empty data"}
    except Exception as e:
        return {"status": "error", "message": f"Feature extraction failed: {e}"}

    # Step 6: Handle Missing Values in Features (NaN)
    try:
        missing_features = feature_data.isnull().sum().sum()
        if missing_features > 0:
            print(f"⚠️ Warning: Found {missing_features} missing values in extracted features.")
            feature_data = feature_data.fillna(0)  # Replace NaNs with 0
            print("✅ NaN values in features replaced with 0.")
    except Exception as e:
        return {"status": "error", "message": f"NaN handling in features failed: {e}"}

    # Step 7: Save Features to CSV
    try:
        feature_data.to_csv(FEATURES_FILE, index=False)
    except Exception as e:
        return {"status": "error", "message": f"Failed to save features CSV: {e}"}

    # Step 8: Perform Prediction
    try:
        expected_features = model.n_features_in_
        actual_features = feature_data.shape[1]

        if actual_features != expected_features:
            return {
                "status": "error",
                "message": f"Feature count mismatch: Model expects {expected_features}, but got {actual_features}"
            }

        predictions = model.predict(feature_data)
        print(f"🔮 Predictions: {predictions.tolist()}")
    except Exception as e:
        return {"status": "error", "message": f"Model prediction failed: {e}"}

    return {"status": "success", "predictions": predictions.tolist()}
