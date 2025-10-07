from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Optional, Literal, Any
from taxipred.utils.constants import (
    MODEL_PATH,
    FEATURE_NAMES_PATH,
    LABEL_ENCODERS_PATH,
    MODEL_METADATA_PATH,
    CLEANED_TAXI_DATA
)

model = None
feature_names = None
label_encoders = None
metadata = None
df_data = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_names, label_encoders, metadata, df_data
    
    try:
        if MODEL_PATH.exists():
            model = joblib.load(str(MODEL_PATH))
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"ERROR: Model not found at {MODEL_PATH}")
        
        if FEATURE_NAMES_PATH.exists():
            feature_names = joblib.load(str(FEATURE_NAMES_PATH))
            print(f"Feature names loaded: {feature_names}")
        else:
            print(f"ERROR: Features not found at {FEATURE_NAMES_PATH}")
        
        if LABEL_ENCODERS_PATH.exists():
            label_encoders = joblib.load(str(LABEL_ENCODERS_PATH))
            print(f"Label encoders loaded: {list(label_encoders.keys())}")
        else:
            print(f"ERROR: Encoders not found at {LABEL_ENCODERS_PATH}")
        
        if MODEL_METADATA_PATH.exists():
            metadata = joblib.load(str(MODEL_METADATA_PATH))
            print(f"Metadata loaded")
        else:
            print(f"ERROR: Metadata not found at {MODEL_METADATA_PATH}")
        
        try:
            df_data = pd.read_csv(str(CLEANED_TAXI_DATA))
            print(f"Data loaded: {df_data.shape[0]} rows")
        except Exception as e:
            print(f"Warning: Could not load data: {e}")
            
    except Exception as e:
        print(f"Error loading resources: {str(e)}")
    
    yield

app = FastAPI(
    title="Taxi Price Prediction API",
    description="API for predicting taxi trip prices",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionInput(BaseModel):
    Trip_Distance_km: float = Field(..., description="Trip distance in kilometers", gt=0, le=1000)
    Time_of_Day: Literal["Morning", "Afternoon", "Evening", "Night"] = Field(
        ..., description="Time of day category"
    )
    Day_of_Week: Literal["Weekday", "Weekend"] = Field(
        ..., description="Weekday or Weekend"
    )
    Passenger_Count: int = Field(..., description="Number of passengers", ge=1, le=8)
    Traffic_Conditions: Literal["Low", "Medium", "High"] = Field(
        ..., description="Traffic condition level"
    )
    Weather: Literal["Clear", "Rain", "Fog", "Snow"] = Field(
        ..., description="Weather condition"
    )
    Base_Fare: float = Field(..., description="Base fare amount", ge=0)
    Per_Km_Rate: float = Field(..., description="Rate per kilometer", ge=0)
    Per_Minute_Rate: float = Field(..., description="Rate per minute", ge=0)
    Trip_Duration_Minutes: float = Field(..., description="Trip duration in minutes", gt=0, le=600)
    
    class Config:
        json_schema_extra = {
            "example": {
                "Trip_Distance_km": 10.5,
                "Time_of_Day": "Morning",
                "Day_of_Week": "Weekday",
                "Passenger_Count": 2,
                "Traffic_Conditions": "Medium",
                "Weather": "Clear",
                "Base_Fare": 5.0,
                "Per_Km_Rate": 2.5,
                "Per_Minute_Rate": 0.5,
                "Trip_Duration_Minutes": 25.0
            }
        }


class PredictionOutput(BaseModel):
    predicted_price: float
    confidence_interval: Dict[str, float]
    input_summary: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    encoders_loaded: bool
    features_count: Optional[int]


def encode_input(input_dict: dict, encoders: dict) -> dict:
    encoded = input_dict.copy()
    
    for feature, encoder in encoders.items():
        if feature in encoded:
            try:
                value = str(encoded[feature])
                encoded[feature] = encoder.transform([value])[0]
            except ValueError as e:
                raise ValueError(f"Invalid value '{encoded[feature]}' for {feature}. "
                               f"Valid values: {list(encoder.classes_)}")
    
    return encoded


@app.get("/")
async def root():
    return {"message": "Taxi Price Prediction API", "status": "running"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if all([model, feature_names, label_encoders]) else "degraded",
        model_loaded=model is not None,
        encoders_loaded=label_encoders is not None,
        features_count=len(feature_names) if feature_names else None
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict_price(input_data: PredictionInput):
    
    if model is None or label_encoders is None or feature_names is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    try:
        input_dict = input_data.model_dump()
        
        encoded_dict = encode_input(input_dict, label_encoders)
        
        feature_values = [encoded_dict[feature] for feature in feature_names]
        
        input_df = pd.DataFrame([feature_values], columns=feature_names)
        
        prediction = model.predict(input_df)[0]
        
        prediction = max(0, prediction)
        
        lower_bound = prediction * 0.80
        upper_bound = prediction * 1.20
        
        return PredictionOutput(
            predicted_price=round(prediction, 2),
            confidence_interval={
                "lower": round(lower_bound, 2),
                "upper": round(upper_bound, 2),
                "range_percent": 20
            },
            input_summary={
                "distance_km": input_data.Trip_Distance_km,
                "duration_min": input_data.Trip_Duration_Minutes,
                "passengers": input_data.Passenger_Count,
                "time": input_data.Time_of_Day,
                "traffic": input_data.Traffic_Conditions
            }
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/features")
async def get_features():
    
    if not all([feature_names, label_encoders, metadata]):
        raise HTTPException(status_code=503, detail="Model metadata not loaded")
    
    categorical_info = {}
    for feature, encoder in label_encoders.items():
        categorical_info[feature] = {
            "type": "categorical",
            "valid_values": list(encoder.classes_)
        }
    
    numeric_features = [f for f in feature_names if f not in label_encoders]
    
    return {
        "total_features": len(feature_names),
        "numeric_features": numeric_features,
        "categorical_features": categorical_info,
        "model_type": metadata.get('model_type', 'Unknown')
    }


@app.get("/sample")
async def get_sample_prediction():
    
    if df_data is None:
        sample_data = PredictionInput(
            Trip_Distance_km=10.5,
            Time_of_Day="Morning",
            Day_of_Week="Weekday",
            Passenger_Count=2,
            Traffic_Conditions="Medium",
            Weather="Clear",
            Base_Fare=5.0,
            Per_Km_Rate=2.5,
            Per_Minute_Rate=0.5,
            Trip_Duration_Minutes=25.0
        )
    else:
        sample_row = df_data.sample(n=1).iloc[0]
        sample_data = PredictionInput(
            Trip_Distance_km=float(sample_row['Trip_Distance_km']),
            Time_of_Day=str(sample_row['Time_of_Day']),
            Day_of_Week=str(sample_row['Day_of_Week']),
            Passenger_Count=int(sample_row['Passenger_Count']),
            Traffic_Conditions=str(sample_row['Traffic_Conditions']),
            Weather=str(sample_row['Weather']),
            Base_Fare=float(sample_row['Base_Fare']),
            Per_Km_Rate=float(sample_row['Per_Km_Rate']),
            Per_Minute_Rate=float(sample_row['Per_Minute_Rate']),
            Trip_Duration_Minutes=float(sample_row['Trip_Duration_Minutes'])
        )
    
    result = await predict_price(sample_data)
    
    return {
        "sample_input": sample_data.model_dump(),
        "prediction": result
    }


@app.get("/stats")
async def get_stats():
    
    if df_data is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    numeric_cols = df_data.select_dtypes(include=[np.number]).columns.tolist()
    
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "mean": float(df_data[col].mean()),
            "median": float(df_data[col].median()),
            "min": float(df_data[col].min()),
            "max": float(df_data[col].max())
        }
    
    return {
        "total_rows": len(df_data),
        "statistics": stats
    }