from importlib.resources import files

TAXI_CSV_PATH = files("taxipred").joinpath("data/taxi_trip_pricing.csv")
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "final_model_full.joblib"
)

CLEANED_TAXI_DATA = files("taxipred").joinpath("processed_data/data_cleaned.csv")
