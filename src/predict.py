import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

#                                                           <- Configs from your previous scripts
CLASSIFIER_PATH = Path("./models/classifier.json")
REGRESSOR_PATH  = Path("./models/regressor.json")
OUTPUT_DIR      = Path("./predictions_2026")
OUTPUT_DIR.mkdir(exist_ok=True)


LAT_RANGE = (25.61, 26.63)    #                              <- crop coordinates
LON_RANGE = (-98.78, -96.77) 
RESOLUTION = 0.1     #                                       <- IMERG resolution

FEATURES = [
    'lat', 'lon',
    'month_sin', 'month_cos',
    'doy_sin', 'doy_cos',
    'day_of_month', 'week_of_year',
    'precip_lag1', 'precip_lag3', 'precip_lag7',
    'precip_lag14', 'precip_lag30',
    'precip_roll7', 'precip_roll30',
    'precip_max7', 'precip_max30'
]

def generate_2026_grid():
    print("Generating 2026 RGV coordinate grid...")
    lats = np.arange(LAT_RANGE[0], LAT_RANGE[1], RESOLUTION)
    lons = np.arange(LON_RANGE[0], LON_RANGE[1], RESOLUTION)
    dates = pd.date_range(start="2026-01-01", end="2026-12-31", freq='D')
    
    grid = []
    for date in dates:
        for lat in lats:
            for lon in lons:
                grid.append({'time': date, 'lat': lat, 'lon': lon})
    
    df = pd.DataFrame(grid)
    
    #                                                                               <- Seasonal Encoding (similar to preprocess.py)
    print("Encoding seasonal features...")
    df['month'] = df['time'].dt.month
    df['day_of_year'] = df['time'].dt.dayofyear
    df['day_of_month'] = df['time'].dt.day
    df['week_of_year'] = df['time'].dt.isocalendar().week.astype(int)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin']   = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos']   = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    #                                                                      <- Lag Simulation: Since theres no 2025 data, we initialize with historical "typical" values (e.g., 0.5mm) so the model has non-zero inputs
    for lag in [1, 3, 7, 14, 30]:
        df[f'precip_lag{lag}'] = 0.5 
    for roll in [7, 30]:
        df[f'precip_roll{roll}'] = 0.5
    for m in [7, 30]:
        df[f'precip_max{m}'] = 1.0

    return df

def run_2026_prediction(threshold=0.50):
    if not CLASSIFIER_PATH.exists() or not REGRESSOR_PATH.exists():
        print("Error: Models not found in /models/ directory.")
        return

    #                                                                                                       <- Load Models
    clf = xgb.XGBClassifier()
    clf.load_model(CLASSIFIER_PATH)
    reg = xgb.XGBRegressor()
    reg.load_model(REGRESSOR_PATH)

    #                                                                                                       <- Generating Data
    df_2026 = generate_2026_grid()
    X = df_2026[FEATURES]

    #                                                                                                       <- S1: Classifier
    print(f"Running Stage 1 Classifier (Threshold: {threshold})...")
    probs = clf.predict_proba(X)[:, 1]
    df_2026['rain_probability'] = probs
    df_2026['will_rain'] = (probs >= threshold).astype(int)

    #                                                                                                       <- S2: Regressor
    print("Running Stage 2 Regressor...")
    df_2026['predicted_mm'] = 0.0
    rainy_mask = df_2026['will_rain'] == 1
    
    if rainy_mask.any():
        preds_log = reg.predict(X[rainy_mask])
        df_2026.loc[rainy_mask, 'predicted_mm'] = np.expm1(preds_log)

    
    output_path = OUTPUT_DIR / "rgv_2026_projections.parquet"
    df_2026.to_parquet(output_path, index=False)
    print(f"Saved projections to {output_path}")

    #                                                                                                        <- Monthly Forecast vizualization
    monthly_summary = df_2026.groupby(df_2026['time'].dt.month)['predicted_mm'].mean()
    plt.figure(figsize=(10, 5))
    plt.bar(monthly_summary.index, monthly_summary.values, color='teal', alpha=0.7)
    plt.title("Projected Average Daily Precipitation for RGV (2026)")
    plt.xlabel("Month")
    plt.ylabel("Avg mm/day")
    plt.xticks(range(1, 13), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(OUTPUT_DIR / "seasonal_forecast_2026.png")
    print("Generated seasonal forecast plot.")

if __name__ == "__main__":
    run_2026_prediction(threshold=0.45)                                                                     # Using your preferred threshold