import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
FINAL_DIR = Path("./data/final_training_set")
MODEL_DIR  = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [ #                                                                   <- Features
    'lat', 'lon',
    'month_sin', 'month_cos',
    'doy_sin', 'doy_cos',
    'day_of_month', 'week_of_year',
    'precip_lag1', 'precip_lag3', 'precip_lag7',
    'precip_lag14', 'precip_lag30',
    'precip_roll7', 'precip_roll30',
    'precip_max7', 'precip_max30'
]

#                                                                               90th percentile threshold for sample weighting
EXTREME_WEIGHT    = 4.0   #                                                     <- extreme events are worth 4x in loss
EXTREME_THRESHOLD = 0.90  #                                                     <- top 10% of rainy days get boosted weight

def train_xgboost():
    print("Loading full dataset...")
    df = pd.read_parquet(FINAL_DIR / "final_dataset.parquet")

    print(f"Total rows:  {len(df):,}")
    print(f"Year range:  {int(df['year'].min())} to {int(df['year'].max())}")

    target_col     = next(c for c in ['precipitationCal', 'precipitation'] if c in df.columns)
    target_log_col = f'{target_col}_log'

    df['rained'] = (df[target_col] > 0).astype(int)  #                               <- Binary rain label for classifier

    
    train_df = df[df['year'] <= 2021]  #                                             <- Training split yrs
    test_df  = df[df['year'] >= 2022]  #                                             <- Test split yrs

    print(f"Training set: {len(train_df):,} rows")
    print(f"Test set:     {len(test_df):,} rows")
    print(f"Rain rate:    {df['rained'].mean()*100:.1f}% of days")

    #                                                                               <- Stage 1: Classifier (rain / no-rain)
    print("\n Stage 1: Training rain/no-rain classifier ")

    X_train_clf = train_df[FEATURES]
    y_train_clf = train_df['rained']
    X_test_clf  = test_df[FEATURES]
    y_test_clf  = test_df['rained']

    #                                                                                  <- scale_pos_weight balances class imbalance
    rain_rate = y_train_clf.mean()
    scale_pos = (1 - rain_rate) / rain_rate

    classifier = xgb.XGBClassifier( #                                                  <- classifier settings
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        tree_method='hist',
        device='cpu',
        early_stopping_rounds=50,
        eval_metric='logloss'
    )

    classifier.fit(
        X_train_clf, y_train_clf,
        eval_set=[(X_test_clf, y_test_clf)],
        verbose=100
    )

    classifier.save_model(MODEL_DIR / "classifier.json")
    print("Classifier saved to models/classifier.json")

    #                                                                                    Stage 2: Regressor (amount on rainy days only)
    print("\nStage 2: Training precipitation amount regressor ")

    train_rainy = train_df[train_df[target_col] > 0].copy()
    test_rainy  = test_df[test_df[target_col] > 0].copy()

    print(f"Rainy training rows: {len(train_rainy):,}")
    print(f"Rainy test rows:     {len(test_rainy):,}")

    #                                                                                   Sample weights useful boost extreme precipitation events
    #                                                                                   Anything above 90th percentile gets EXTREME_WEIGHT, rest get 1.0
    p90 = train_rainy[target_col].quantile(EXTREME_THRESHOLD)
    print(f"90th percentile threshold: {p90:.2f} mm/day — events above this get {EXTREME_WEIGHT}x weight")

    sample_weights = np.where(train_rainy[target_col] >= p90, EXTREME_WEIGHT, 1.0)

    X_train_reg = train_rainy[FEATURES]
    y_train_reg = train_rainy[target_log_col]
    X_test_reg  = test_rainy[FEATURES]
    y_test_reg  = test_rainy[target_log_col]

    regressor = xgb.XGBRegressor(    #                                                  <- Regressor setting
        objective='reg:tweedie',
        tweedie_variance_power=1.7,  #                                                  <- increased from 1.5  more emphasis on large values
        n_estimators=3000,
        max_depth=8,                 #                                                  <- deep enough to capture extreme patterns
        learning_rate=0.02,          #                                                  <- low LR + high n_estimators = better generalization
        subsample=0.6,
        colsample_bytree=0.7,
        min_child_weight=1,          #                                                  <- allows small leaf nodes for extremes
        tree_method='hist',
        device='cpu',
        early_stopping_rounds=50
    )

    regressor.fit(
        X_train_reg, y_train_reg,
        sample_weight=sample_weights,       #                                           <- sample weights on extreme events
        eval_set=[(X_test_reg, y_test_reg)],
        verbose=100
    )

    regressor.save_model(MODEL_DIR / "regressor.json")
    print("Regressor saved to models/regressor.json")
    print("\nTwo-stage training complete.")

if __name__ == "__main__":
    train_xgboost()