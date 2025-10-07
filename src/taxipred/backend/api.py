from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI()

# Load model
try:
    API_DIR = Path(__file__).resolve().parent  # backend/
    TAXIPRED_DIR = API_DIR.parent  # taxipred/
    MODEL_PATH = TAXIPRED_DIR / 'models' / 'final_model_full.joblib'

    print(f"API dir: {API_DIR}")
    print(f"Looking for model at: {MODEL_PATH}")
    print(f"File exists: {MODEL_PATH.exists()}")

    if not MODEL_PATH.exists():
        # List what's actually in the models folder
        models_dir = TAXIPRED_DIR / 'models'
        if models_dir.exists():
            print(f"Files in models/: {list(models_dir.iterdir())}")
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    package = joblib.load(MODEL_PATH)
    
    # Check if it's a package dict or just the model
    if isinstance(package, dict):
        model = package.get('model')
        encoders = package.get('label_encoders', {})
        feature_names = package.get('feature_names', [])
        encoding_type = package.get('encoding_type', 'unknown')
        model_name = package.get('model_name', 'Unknown')
    else:
        # If it's just the model directly
        model = package
        encoders = {}
        feature_names = []
        encoding_type = 'unknown'
        model_name = type(model).__name__
    
    print(f"Model loaded: {model_name}")
    print(f"Encoding: {encoding_type}")
    
except Exception as e:
    print(f"ERROR loading model: {str(e)}")
    raise

class TripInput(BaseModel):
    Trip_Distance_km: float
    Trip_Duration_Minutes: float
    Day_of_Week: str
    Time_of_Day: str
    Passenger_Count: int
    Base_Fare: float
    Per_Km_Rate: float
    Per_Minute_Rate: float
    Traffic_Conditions: str
    Weather: str

@app.post("/taxi/predict")
def predict_price(trip: TripInput):
    try:
        # Feature engineering
        price_per_km = trip.Base_Fare / max(trip.Trip_Distance_km, 0.1)
        price_per_minute = trip.Base_Fare / max(trip.Trip_Duration_Minutes, 1)
        speed_kmh = (trip.Trip_Distance_km / max(trip.Trip_Duration_Minutes, 1)) * 60
        is_weekend = 1 if trip.Day_of_Week in ['Saturday', 'Sunday'] else 0
        is_rush_hour = 1 if trip.Time_of_Day in ['Morning', 'Evening'] else 0
        
        # If using label encoding
        if encoders:
            time_encoded = encoders['Time_of_Day'].transform([trip.Time_of_Day])[0]
            day_encoded = encoders['Day_of_Week'].transform([trip.Day_of_Week])[0]
            traffic_encoded = encoders['Traffic_Conditions'].transform([trip.Traffic_Conditions])[0]
            weather_encoded = encoders['Weather'].transform([trip.Weather])[0]
            
            data = pd.DataFrame([{
                'Trip_Distance_km': trip.Trip_Distance_km,
                'Passenger_Count': trip.Passenger_Count,
                'Base_Fare': trip.Base_Fare,
                'Per_Km_Rate': trip.Per_Km_Rate,
                'Per_Minute_Rate': trip.Per_Minute_Rate,
                'Trip_Duration_Minutes': trip.Trip_Duration_Minutes,
                'Time_of_Day_Encoded': time_encoded,
                'Day_of_Week_Encoded': day_encoded,
                'Traffic_Conditions_Encoded': traffic_encoded,
                'Weather_Encoded': weather_encoded,
                'Price_Per_Km': price_per_km,
                'Price_Per_Minute': price_per_minute,
                'Speed_kmh': speed_kmh,
                'Is_Weekend': is_weekend,
                'Is_Rush_Hour': is_rush_hour
            }])
        else:
            # If no encoders available, create basic DataFrame
            data = pd.DataFrame([{
                'Trip_Distance_km': trip.Trip_Distance_km,
                'Passenger_Count': trip.Passenger_Count,
                'Base_Fare': trip.Base_Fare,
                'Per_Km_Rate': trip.Per_Km_Rate,
                'Per_Minute_Rate': trip.Per_Minute_Rate,
                'Trip_Duration_Minutes': trip.Trip_Duration_Minutes,
            }])
        
        # Align columns if feature_names available
        if feature_names:
            data = data.reindex(columns=feature_names, fill_value=0)
        
        # Predict
        prediction = model.predict(data)[0]
        
        return {"price": round(float(prediction), 2)}
        
    except Exception as e:
        import traceback
        print("ERROR:", str(e))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Taxi Price Prediction API", "status": "running"}

print("Day_of_Week classes:", encoders['Day_of_Week'].classes_)
print("Time_of_Day classes:", encoders['Time_of_Day'].classes_)
print("Traffic classes:", encoders['Traffic_Conditions'].classes_)
print("Weather classes:", encoders['Weather'].classes_)