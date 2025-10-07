from importlib.resources import files
from pathlib import Path

# Data paths using importlib.resources
TAXI_CSV_PATH = files("taxipred").joinpath("data/taxi_trip_pricing.csv")
CLEANED_TAXI_DATA = files("taxipred").joinpath("processed_data/data_cleaned.csv")

# Model paths using Path
TAXIPRED_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = TAXIPRED_ROOT / "models" / "taxi_price_model.pkl"
FEATURE_NAMES_PATH = TAXIPRED_ROOT / "models" / "feature_names.pkl"
LABEL_ENCODERS_PATH = TAXIPRED_ROOT / "models" / "label_encoders.pkl"
MODEL_METADATA_PATH = TAXIPRED_ROOT / "models" / "model_metadata.pkl"