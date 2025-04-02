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
PUSHBULLET_API_KEY = "o.vfCOlEsqdU8eQBSmYSWwijLc6puBeQtf"
PUSHBULLET_API_URL = "https://api.pushbullet.com/v2/pushes"

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

# Function to send pushbullet notification
def send_pushbullet_notification(title, message):
    """Sends a notification to Pushbullet."""
    headers = {
        "Access-Token": PUSHBULLET_API_KEY,
        "Content-Type": "application/json"
    }
    
    data = {
        "type": "note",
        "title": title,
        "body": message
    }
    
    try:
        response = requests.post(PUSHBULLET_API_URL, headers=headers, json=data)
        if response.status_code == 200:
            print("✅ Notification sent successfully!")
            return True
        else:
            print(f"❌ Failed to send notification: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error sending notification: {str(e)}")
        return False

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

# # Function to run prediction
# async def predict_fall(csv_file):
#     """Runs ML prediction on the specified CSV file."""
#     try:
#         print(f"🔍 Analyzing file: {csv_file}")
        
#         # Step 1: Load CSV
#         df = pd.read_csv(csv_file)
#         print(f"📊 Loaded Data Shape: {df.shape}")
        
#         if df.empty:
#             error_msg = "CSV file is empty"
#             send_pushbullet_notification("Fall Detection Error", error_msg)
#             return {"status": "error", "message": error_msg}
            
#         # Step 2: Check peak acceleration
#         acceleration_magnitude = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
#         peak_acceleration = acceleration_magnitude.max()
#         print(f"⚡ Peak Acceleration: {peak_acceleration} m/s²")
        
#         if peak_acceleration < 15:
#             message = f"Low peak acceleration detected ({peak_acceleration:.2f} m/s²). Classified as non-fall."
#             send_pushbullet_notification("Fall Detection Result", message)
#             return {"status": "success", "prediction": "non_fall", "reason": "low_acceleration"}
            
#         # Step 3: Handle Missing Values (NaN)
#         missing_values = df.isnull().sum().sum()
#         if missing_values > 0:
#             print(f"⚠ Warning: Found {missing_values} missing values in data.")
#             df = df.fillna(df.mean())  # Impute missing values with column mean
        
#         # Step 4: Apply Z-score Normalization
#         sensor_columns = ['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz']
#         df[sensor_columns] = df[sensor_columns].apply(zscore)
        
#         # Step 5: Extract Features
#         feature_data = extract_features_sliding_window(df)
#         print(f"🧩 Extracted Feature Shape: {feature_data.shape}")
        
#         if feature_data.empty:
#             error_msg = "Feature extraction returned empty data"
#             send_pushbullet_notification("Fall Detection Error", error_msg)
#             return {"status": "error", "message": error_msg}
            
#         # Step 6: Handle Missing Values in Features
#         missing_features = feature_data.isnull().sum().sum()
#         if missing_features > 0:
#             print(f"⚠ Warning: Found {missing_features} missing values in extracted features.")
#             feature_data = feature_data.fillna(0)  # Replace NaNs with 0
            
#         # Step 7: Save Features to CSV
#         feature_data.to_csv(FEATURES_FILE, index=False)
        
#         # Step 8: Perform Prediction
#         expected_features = model.n_features_in_
#         actual_features = feature_data.shape[1]
        
#         if actual_features != expected_features:
#             error_msg = f"Feature count mismatch: Model expects {expected_features}, but got {actual_features}"
#             send_pushbullet_notification("Fall Detection Error", error_msg)
#             return {"status": "error", "message": error_msg}
        
#         predictions = model.predict(feature_data)
#         print(f"🔮 Raw Predictions: {predictions.tolist()}")
        
#         # Prioritize Falls if Detected
#         fall_types = ["forward_fall", "backward_fall", "lateral_fall"]
#         result = "non_fall"
        
#         for fall in fall_types:
#             if fall in predictions:
#                 result = fall
#                 break
        
#         # Send notification based on result
#         if result != "non_fall":
#             message = f"⚠ ALERT! {result.replace('_', ' ').title()} detected! Peak acceleration: {peak_acceleration:.2f} m/s²"
#         else:
#             message = f"Normal activity detected. Peak acceleration: {peak_acceleration:.2f} m/s²"
            
#         send_pushbullet_notification("Fall Detection Result", message)
#         return {"status": "success", "prediction": result, "peak_acceleration": float(peak_acceleration)}
        
#     except Exception as e:
#         error_msg = f"Prediction failed: {str(e)}"
#         send_pushbullet_notification("Fall Detection Error", error_msg)
#         return {"status": "error", "message": error_msg}

from scipy.signal import butter, filtfilt

# Function to create a Butterworth low-pass filter
def create_lowpass_filter(sampling_rate, cutoff=5, order=4):
    nyquist = sampling_rate / 2
    return butter(order, cutoff / nyquist, btype='low')

# Function to run prediction
async def predict_fall(csv_file):
    """Runs ML prediction on the specified CSV file."""
    try:
        print(f"🔍 Analyzing file: {csv_file}")
        
        # Step 1: Load CSV
        df = pd.read_csv(csv_file)
        print(f"📊 Loaded Data Shape: {df.shape}")
        
        if df.empty:
            error_msg = "CSV file is empty"
            send_pushbullet_notification("Fall Detection Error", error_msg)
            return {"status": "error", "message": error_msg}
            
        # Step 2: Calculate sampling rate
        df['time_diff'] = df['time'].diff().fillna(0)
        sampling_rate = 1 / df['time_diff'].mean()

        # Step 3: Apply low-pass filter to acceleration magnitude
        df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
        b, a = create_lowpass_filter(sampling_rate)
        df['acc_mag_filtered'] = filtfilt(b, a, df['acc_mag'])

        # Step 4: Find peak acceleration after filtering
        peak_acceleration = df['acc_mag_filtered'].max()
        print(f"⚡ Peak Acceleration (Filtered): {peak_acceleration:.2f} m/s²")

        if peak_acceleration < 15:
            message = f"Low peak acceleration detected ({peak_acceleration:.2f} m/s²). Classified as non-fall."
            send_pushbullet_notification("Fall Detection Result", message)
            return {"status": "success", "prediction": "non_fall", "reason": "low_acceleration"}

        # Step 5: Handle Missing Values (NaN)
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"⚠ Warning: Found {missing_values} missing values in data.")
            df = df.fillna(df.mean())  # Impute missing values with column mean
        
        # Step 6: Apply Z-score Normalization
        sensor_columns = ['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz']
        df[sensor_columns] = df[sensor_columns].apply(zscore)
        
        # Step 7: Extract Features
        feature_data = extract_features_sliding_window(df)
        print(f"🧩 Extracted Feature Shape: {feature_data.shape}")
        
        if feature_data.empty:
            error_msg = "Feature extraction returned empty data"
            send_pushbullet_notification("Fall Detection Error", error_msg)
            return {"status": "error", "message": error_msg}
            
        # Step 8: Handle Missing Values in Features
        missing_features = feature_data.isnull().sum().sum()
        if missing_features > 0:
            print(f"⚠ Warning: Found {missing_features} missing values in extracted features.")
            feature_data = feature_data.fillna(0)  # Replace NaNs with 0
            
        # Step 9: Save Features to CSV
        feature_data.to_csv(FEATURES_FILE, index=False)
        
        # Step 10: Perform Prediction
        expected_features = model.n_features_in_
        actual_features = feature_data.shape[1]
        
        if actual_features != expected_features:
            error_msg = f"Feature count mismatch: Model expects {expected_features}, but got {actual_features}"
            send_pushbullet_notification("Fall Detection Error", error_msg)
            return {"status": "error", "message": error_msg}
        
        predictions = model.predict(feature_data)
        print(f"🔮 Raw Predictions: {predictions.tolist()}")
        
        # Step 11: Determine Fall Type
        fall_types = ["forward_fall", "backward_fall", "lateral_fall"]
        result = "non_fall"
        
        for fall in fall_types:
            if fall in predictions:
                result = fall
                break
        
        # Step 12: Send Notification
        if result != "non_fall":
            message = f"⚠ ALERT! {result.replace('_', ' ').title()} detected! Peak acceleration: {peak_acceleration:.2f} m/s²"
        else:
            message = f"Normal activity detected. Peak acceleration: {peak_acceleration:.2f} m/s²"
            
        send_pushbullet_notification("Fall Detection Result", message)
        return {"status": "success", "prediction": result, "peak_acceleration": float(peak_acceleration)}
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        send_pushbullet_notification("Fall Detection Error", error_msg)
        return {"status": "error", "message": error_msg}

@app.post("/sensor")
async def store_sensor_data(sensors: List[SensorData]):
    """Stores incoming sensor data as a CSV file and automatically runs prediction"""
    try:
        # Store data
        new_csv_file = get_new_filename()
        df = pd.DataFrame([sensor.dict() for sensor in sensors])
        df.to_csv(new_csv_file, index=False)
        
        # Remove old CSV files (keep only the latest)
        for file in os.listdir():
            if file.startswith(CSV_BASE_NAME) and file.endswith(CSV_EXTENSION) and file != new_csv_file:
                os.remove(file)
        
        # Automatically run prediction
        prediction_result = await predict_fall(new_csv_file)
        
        return {
            "status": "success", 
            "message": f"Data stored in {new_csv_file}",
            "prediction_result": prediction_result
        }
    
    except Exception as e:
        error_msg = f"Error processing sensor data: {str(e)}"
        send_pushbullet_notification("Fall Detection Error", error_msg)
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
    """Extracts features using a sliding window approach."""
    features = []
    sensor_columns = ['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Bx', 'By', 'Bz']
    
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[start:start + window_size]
        feature_row = {}

        # Extract features for each sensor axis
        for axis in sensor_columns:
            axis_features = extract_features(window, axis)
            feature_row.update(axis_features)
        
        # Compute Additional Global Features
        acc_magnitude = np.sqrt(window['ax']**2 + window['ay']**2 + window['az']**2)
        gyro_magnitude = np.sqrt(window['wx']**2 + window['wy']**2 + window['wz']**2)
        
        # Vertical Impact Ratio (VIR)
        vir = np.mean(np.abs(window['az']) / (acc_magnitude + 1e-6))  # Avoid division by zero
        
        # Angular Momentum Change (AMC)
        amc = np.mean(np.sqrt(np.diff(window['wx'])**2 + np.diff(window['wy'])**2 + np.diff(window['wz'])**2))
        
        # Energy Expenditure Index (EEI)
        eei = np.trapz(acc_magnitude, dx=1)  # Integrate acceleration over time
        
        # Add new global features to the row
        feature_row['vir'] = vir
        feature_row['amc'] = amc
        feature_row['eei'] = eei
        
        features.append(feature_row)

    return pd.DataFrame(features)

@app.get("/predict")
async def run_ml_prediction():
    """Runs ML prediction on the latest stored CSV file and returns output"""
    csv_files = sorted(
        [f for f in os.listdir() if f.startswith(CSV_BASE_NAME) and f.endswith(CSV_EXTENSION)],
        key=lambda x: int(x.split("_")[-1].split(".")[0]),
        reverse=True  
    )

    if not csv_files:
        return {"status": "error", "message": "No CSV file found for prediction"}

    latest_file = csv_files[0]
    return await predict_fall(latest_file)
