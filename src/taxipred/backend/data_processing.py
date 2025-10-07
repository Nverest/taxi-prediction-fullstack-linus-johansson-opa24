import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

def load_and_clean_data(filepath):
    #Load data and perform initial cleaning
    print("Loading data")
    df = pd.read_csv(filepath)
    
    print(f"Initial shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nNull values:\n{df.isnull().sum()}")
    
    df_clean = df.copy()
    
    if 'Trip_Distance_km' in df_clean.columns:
        df_clean['Trip_Distance_km'].fillna(df_clean['Trip_Distance_km'].median(), inplace=True)
    
    if 'Passenger_Count' in df_clean.columns:
        df_clean['Passenger_Count'].fillna(df_clean['Passenger_Count'].mode()[0], inplace=True)
    
    categorical_cols = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']
    for col in categorical_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    rate_cols = ['Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes']
    for col in rate_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    if 'Trip_Price' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['Trip_Price'])
    
    if 'Trip_Price' in df_clean.columns:
        lower_bound = df_clean['Trip_Price'].quantile(0.01)
        upper_bound = df_clean['Trip_Price'].quantile(0.99)
        print(f"\nRemoving outliers: Trip_Price < ${lower_bound:.2f} or > ${upper_bound:.2f}")
        df_clean = df_clean[(df_clean['Trip_Price'] >= lower_bound) & 
                           (df_clean['Trip_Price'] <= upper_bound)]
    
    print(f"\nCleaned shape: {df_clean.shape}")
    print(f"Rows removed: {df.shape[0] - df_clean.shape[0]}")
    
    return df_clean


def prepare_features(df):
    #Prepare features for modeling with categorical encoding
    
    df_model = df.copy()
    
    exclude_cols = ['Trip_Price']
    
    categorical_cols = [col for col in df_model.columns 
                       if col not in exclude_cols and df_model[col].dtype == 'object']
    
    numeric_cols = [col for col in df_model.columns 
                   if col not in exclude_cols and df_model[col].dtype in ['int64', 'float64']]
    
    print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    
    df_encoded = df_model.copy()
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_model[col].astype(str))
        label_encoders[col] = le
        print(f"\n{col} encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    feature_cols = numeric_cols + categorical_cols
    
    X = df_encoded[feature_cols]
    y = df_encoded['Trip_Price']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nTarget statistics:")
    print(f"  Mean price: ${y.mean():.2f}")
    print(f"  Median price: ${y.median():.2f}")
    print(f"  Min price: ${y.min():.2f}")
    print(f"  Max price: ${y.max():.2f}")
    
    return X, y, feature_cols, label_encoders


def train_and_evaluate_models(X, y):
#Train multiple models and compare performance
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=100, 
            max_depth=15, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    results = {}
    print("MODEL EVALUATION")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred_train = pipeline.predict(X_train)
        y_pred_test = pipeline.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        train_mape = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
        test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        
        results[name] = {
            'pipeline': pipeline,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mape': train_mape,
            'test_mape': test_mape
        }
        
        print(f"{name} Results:")
        print(f"  Train MAE: ${train_mae:.2f} | Test MAE: ${test_mae:.2f}")
        print(f"  Train RMSE: ${train_rmse:.2f} | Test RMSE: ${test_rmse:.2f}")
        print(f"  Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}")
        print(f"  Train MAPE: {train_mape:.2f}% | Test MAPE: {test_mape:.2f}%")
    
    best_model_name = min(results, key=lambda x: results[x]['test_mae'])
    print(f"BEST MODEL: {best_model_name}")
    print(f"  Test MAE: ${results[best_model_name]['test_mae']:.2f}")
    print(f"  Test MAPE: {results[best_model_name]['test_mape']:.2f}%")
    
    return results, best_model_name


def train_final_model(X, y, model_type='Random Forest'):
    #Train final model on all data
    
    print(f"\nTraining final {model_type} on all data...")
    
    if model_type == 'Random Forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    else:
        model = RandomForestRegressor(random_state=42)
    
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    final_pipeline.fit(X, y)
    
    y_pred = final_pipeline.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    print(f"\nFinal Model Performance (on all data):")
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R2: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return final_pipeline


def save_model(pipeline, feature_cols, label_encoders, models_dir, model_name='taxi_price_model'):
    #Save the trained model and metadata
    
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / f'{model_name}.pkl'
    joblib.dump(pipeline, str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    feature_path = models_dir / 'feature_names.pkl'
    joblib.dump(feature_cols, str(feature_path))
    print(f"Feature names saved to: {feature_path}")
    
    encoders_path = models_dir / 'label_encoders.pkl'
    joblib.dump(label_encoders, str(encoders_path))
    print(f"Label encoders saved to: {encoders_path}")
    
    metadata = {
        'feature_columns': feature_cols,
        'model_type': type(pipeline.named_steps['model']).__name__,
        'n_features': len(feature_cols),
        'categorical_features': list(label_encoders.keys()),
        'numeric_features': [col for col in feature_cols if col not in label_encoders.keys()]
    }
    metadata_path = models_dir / 'model_metadata.pkl'
    joblib.dump(metadata, str(metadata_path))
    print(f"Metadata saved to: {metadata_path}")
    
    return model_path


if __name__ == "__main__":
    
    current_file = Path(__file__).resolve()
    print(f"\nCurrent script: {current_file}")

    taxipred_root = current_file.parent.parent
    
    print(f"Taxipred root: {taxipred_root}")
    
    data_path = taxipred_root / "data" / "taxi_trip_pricing.csv"
    models_dir = taxipred_root / "models"
    cleaned_data_path = taxipred_root / "processed_data" / "data_cleaned.csv"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    (cleaned_data_path.parent).mkdir(parents=True, exist_ok=True)
    
    print(f"\nPaths:")
    print(f"  Data: {data_path}")
    print(f"  Models: {models_dir}")
    print(f"  Cleaned: {cleaned_data_path}")
    
    if not data_path.exists():
        print(f"\nERROR: CSV file not found at {data_path}")
        
        print("\nSearching for CSV files in project...")
        project_root = taxipred_root.parent.parent
        csv_files = list(project_root.rglob("taxi_trip_pricing.csv"))
        
        if csv_files:
            print(f"\nFound file at: {csv_files[0]}")
            data_path = csv_files[0]
            print(f"Using: {data_path}")
        else:
            print("\nNo CSV file found in project")
            exit(1)
    else:
        print(f"\nData file found")
    
    df_clean = load_and_clean_data(str(data_path))
    
    df_clean.to_csv(str(cleaned_data_path), index=False)
    print(f"\nCleaned data saved to: {cleaned_data_path}")
    
    X, y, feature_cols, label_encoders = prepare_features(df_clean)

    results, best_model_name = train_and_evaluate_models(X, y)
    
    final_pipeline = train_final_model(X, y, model_type=best_model_name)
    
    model_path = save_model(final_pipeline, feature_cols, label_encoders, models_dir)
    
    print(f"\nModel files created:")
    print(f"  1. {models_dir / 'taxi_price_model.pkl'}")
    print(f"  2. {models_dir / 'feature_names.pkl'}")
    print(f"  3. {models_dir / 'label_encoders.pkl'}")
    print(f"  4. {models_dir / 'model_metadata.pkl'}")
    print(f"\nData files:")