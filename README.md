# Taxi Price Prediction - Full Stack Application

This has been a school project implementing a complete machine learning solution for predicting taxi fares. This project demonstrates end-to-end ML workflow including data processing, model training, API development, and interactive frontend.

## Project Overview

This application predicts taxi trip prices based on various factors such as distance, duration, traffic conditions, weather, and time of day. Built as a full-stack solution with ML backend and interactive Streamlit frontend.

## Dataset

* **Source** : Kaggle - Taxi Trip Pricing Dataset
* **File** : `taxi_trip_pricing.csv`
* **Features** :
* Trip_Distance_km
* Trip_Duration_Minutes
* Day_of_Week (Weekday/Weekend)
* Time_of_Day (Morning/Afternoon/Evening/Night)
* Passenger_Count
* Traffic_Conditions (Low/Medium/High)
* Weather (Clear/Rain/Fog/Snow)
* Base_Fare
* Per_Km_Rate
* Per_Minute_Rate
* **Target** : Trip_Price

## Technologies Used

### Machine Learning

* **scikit-learn** : Model training and evaluation
* **pandas** : Data manipulation
* **numpy** : Numerical computations
* **joblib** : Model serialization

### Backend

* **FastAPI** : REST API framework
* **Pydantic** : Data validation
* **uvicorn** : ASGI server

### Frontend

* **Streamlit** : Interactive web interface
* **Folium** : Map visualization
* **requests** : API communication

## Project Structure

```
taxi-prediction-fullstack/
├── src/
│   └── taxipred/
│       ├── backend/
│       │   ├── data_processing.py    # ML training pipeline
│       │   └── api.py           # FastAPI application
│       ├── frontend/
│       │   ├── dashboard.py          # Streamlit app
│       │   └── dashboard_util.py     # Helper functions
│       ├── data/
│       │   └── taxi_trip_pricing.csv # Original dataset
│       ├── processed_data/
│       │   └── data_cleaned.csv      # Cleaned dataset
│       ├── models/
│       │   ├── taxi_price_model.pkl  # Trained model
│       │   ├── feature_names.pkl     # Feature list
│       │   ├── label_encoders.pkl    # Categorical encoders
│       │   └── model_metadata.pkl    # Model info
│       └── utils/
│           └── constants.py          # Path configurations
├── visual_asset/
│   └── taxi.png                      # Background image
├── .env                              # Environment variables
├── .gitignore
├── README.md
└── setup.py
```

## Machine Learning Pipeline

### 1. Data Cleaning

* Handle missing values (median for numeric, mode for categorical)
* Remove outliers (1st and 99th percentile for Trip_Price)
* Save cleaned data to `processed_data/`

### 2. Feature Engineering

* Separate numeric and categorical features
* Label encoding for categorical variables (Time_of_Day, Day_of_Week, Traffic_Conditions, Weather)
* StandardScaler for feature normalization

### 3. Model Training

Tested multiple algorithms:

* Linear Regression
* Ridge Regression
* Random Forest Regressor
* Gradient Boosting Regressor

 **Best Model** : Selected based on lowest Mean Absolute Error (MAE)

### 4. Model Evaluation Metrics

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score
* MAPE (Mean Absolute Percentage Error)

### 5. Model Persistence

* Complete sklearn Pipeline saved (includes scaler + model)
* Feature names saved for consistent ordering
* Label encoders saved for categorical variable handling
* Metadata saved for model information

## API Endpoints

### Health Check

```
GET /health
```

Returns API status and model loading information.

### Predict Price

```
POST /predict
```

 **Request Body** :

```json
{
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
```

 **Response** :

```json
{
  "predicted_price": 29.52,
  "confidence_interval": {
    "lower": 23.62,
    "upper": 35.42,
    "range_percent": 20
  },
  "input_summary": {
    "distance_km": 10.5,
    "duration_min": 25.0,
    "passengers": 2,
    "time": "Morning",
    "traffic": "Medium"
  }
}
```

### Feature Information

```
GET /features
```

Returns information about model features and valid categorical values.

### Sample Prediction

```
GET /sample
```

Returns a sample prediction using random data from the dataset.

### Statistics

```
GET /stats
```

Returns dataset statistics.

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/Nverest/taxi-prediction-fullstack-linus-johansson-opa24.git
```

2. **Create virtual environment**

```bash
uv venv
source .venv\Scripts\activate # Mac: source .venv/bin/activate
```

3. **Install dependencies**

```bash
uv pip install -r requirements.txt
```

## Run current version

Will initiate API and streamlit

```
./initiate.ch
```

## Usage

### 1. Train the Model

Run the data processing script to train the model:

```bash
python src/taxipred/backend/data_processing.py
```

This will:

* Load and clean the data
* Train multiple models
* Select the best model
* Save model files to `src/taxipred/models/`
* Save cleaned data to `src/taxipred/processed_data/`

### 2. Start the API Server

```bash
cd src/taxipred/backend
uvicorn api_main:app --reload --port 8000
```

API will be available at: `http://127.0.0.1:8000`
Interactive docs at: `http://127.0.0.1:8000/docs`

### 3. Run the Frontend

In a new terminal:

```bash
streamlit run src/taxipred/frontend/dashboard.py
```

Frontend will open in your browser at: `http://localhost:8501`

## Features

### Frontend Capabilities

* **Address Search** : Search for start and destination addresses
* **Route Visualization** : Interactive map showing the trip route
* **Real-time Predictions** : Get instant price estimates
* **Trip Details** : View distance, duration, and other trip information
* **Configurable Parameters** : Adjust traffic, weather, and passenger count

### API Capabilities

* **Input Validation** : Pydantic models ensure data quality
* **Error Handling** : Comprehensive error messages
* **CORS Support** : Frontend-backend communication enabled
* **Model Loading** : Efficient startup with cached model loading
* **Encoding Management** : Automatic categorical variable encoding

## Model Performance

The model achieves the following performance metrics (example):

* **Test MAE** : ~$X.XX
* **Test RMSE** : ~$X.XX
* **Test R²** : ~0.XX
* **Test MAPE** : ~X.X%

(Actual values depend on your training results)

## Key Learnings

1. **End-to-End ML Pipeline** : From raw data to deployed application
2. **Model Serialization** : Proper saving and loading of ML models
3. **API Development** : Building REST APIs with FastAPI
4. **Frontend Integration** : Connecting ML models with user interfaces
5. **Data Validation** : Using Pydantic for robust input validation
6. **Categorical Encoding** : Handling non-numeric features properly

## Challenges Solved

1. **Path Management** : Using constants for consistent file paths
2. **Categorical Variables** : Proper encoding/decoding with LabelEncoder
3. **Model Loading** : Efficient startup with lifespan events
4. **Feature Ordering** : Maintaining consistent feature order for predictions
5. **CORS Issues** : Enabling frontend-backend communication

## Future Improvements

* Add user authentication
* Implement model versioning
* Add more visualizations (feature importance, prediction distributions)
* Deploy to cloud platform (AWS, Azure, GCP)
* Add A/B testing for different models
* Implement caching for faster predictions
* Add historical trip data tracking
* Integrate real-time traffic data

## Testing

Test the API manually:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## Author

Linus Johansson
LIA Student - Data Engineering/Machine Learning
2024-2026

---

 **Note** : This project demonstrates practical application of machine learning, API development, and full-stack integration skills learned during the course.
