import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
INPUT_DIR = Path(os.getenv("DATA_PROCESSED_DIR", "./data/processed"))
FINAL_DIR = Path("./data/final_training_set")
FINAL_DIR.mkdir(parents=True, exist_ok=True)

def run_preprocessing():
    files = sorted(list(INPUT_DIR.glob("*.parquet")))

    if not files:
        print("No parquet files found in processed directory.")
        return

    print(f"Loading {len(files)} daily files...")

    #                                                                                               <- Load all files at once  (needed for lag features to work across time)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"Total rows loaded: {len(df):,}")

    #                                                                                               <- Find target column
    target_col = next((c for c in ['precipitationCal', 'precipitation'] if c in df.columns), None)
    if not target_col:
        print("No target column found. Exiting.")
        return

    #                                                                   <- Filter negatives
    df = df[df[target_col] >= 0].copy()

    #                                                                   <- Log transform target (compresses extreme values) handles zero inflation log1p(x) = log(x+1) so zeros stay zero
    df[f'{target_col}_log'] = np.log1p(df[target_col])

    # Ensure time is datetime
    df['time'] = pd.to_datetime(df['time'])

    #                                                                   <- Sorting for correct lag ordering
    df = df.sort_values(['lat', 'lon', 'time']).reset_index(drop=True)

    # Season encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['doy_sin']   = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos']   = np.cos(2 * np.pi * df['day_of_year'] / 365)

    #                                                                   <- lag features
    print("Computing lag features...")
    grouped = df.groupby(['lat', 'lon'])[target_col]
    for lag in [1, 3, 7, 14, 30]:
        df[f'precip_lag{lag}'] = grouped.shift(lag)

    #                                                                   <-  Rolling averages
    print("Computing rolling averages...")
    shifted = df.groupby(['lat', 'lon'])[target_col].shift(1)
    df['precip_roll7']  = shifted.rolling(7,  min_periods=1).mean()
    df['precip_roll30'] = shifted.rolling(30, min_periods=1).mean()

    #                                                                                                       <- Max precipitation in past 7 and 30 days
    df['precip_max7']  = df.groupby(['lat', 'lon'])[target_col].shift(1).rolling(7,  min_periods=1).max()
    df['precip_max30'] = df.groupby(['lat', 'lon'])[target_col].shift(1).rolling(30, min_periods=1).max()

    #                                                                                                       <- Droping rows with NaN lags (first 30 days per grid point)
    df = df.dropna()
    print(f"Rows after dropping NaN lags: {len(df):,}")

    #                                                                                                       <- Save full dataset (all days including zeros)
    output_path = FINAL_DIR / "final_dataset.parquet"
    df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
    print(f"Saved full dataset to {output_path}")

    #                                                                                 <- Save only rainy days dataset  zeros dominate and hurt R² in regression so this is key to deal with them
    df_rainy = df[df[target_col] > 0].copy()
    rainy_path = FINAL_DIR / "final_dataset_rainy.parquet"
    df_rainy.to_parquet(rainy_path, index=False, engine='pyarrow', compression='snappy')
    print(f"Saved rainy dataset to {rainy_path}")
    print(f"Rainy days: {len(df_rainy):,} / {len(df):,} total ({100 * len(df_rainy) / len(df):.1f}%)")

    print(f"Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    run_preprocessing()