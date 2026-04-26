import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    accuracy_score, classification_report,
    f1_score, confusion_matrix
)
from pathlib import Path

FINAL_DIR       = Path("./data/final_training_set")
CLASSIFIER_PATH = Path("./models/classifier.json")
REGRESSOR_PATH  = Path("./models/regressor.json")

FEATURES = [   #                                                                         <- Features
    'lat', 'lon',
    'month_sin', 'month_cos',
    'doy_sin', 'doy_cos',
    'day_of_month', 'week_of_year',
    'precip_lag1', 'precip_lag3', 'precip_lag7',
    'precip_lag14', 'precip_lag30',
    'precip_roll7', 'precip_roll30',
    'precip_max7', 'precip_max30'
]

LOG_CLIP = 7.0  #                            log1p(408) ≈ 6.01  clip above max observed
#                                                                                          <- Threshold control 
#                                                                                             Set to None to automatically fine best recall threshold above PRECISION_FLOOR
RAIN_THRESHOLD  = None  #                                                                       Set to a float like 0.35 to manually override
PRECISION_FLOOR = 0.40  #                                                                     only used when RAIN_THRESHOLD is None


def evaluate_model():
    if not CLASSIFIER_PATH.exists() or not REGRESSOR_PATH.exists():
        print("Model files not found. Run model.py first.")
        return

    print("Loading models...")
    classifier = xgb.XGBClassifier()
    classifier.load_model(CLASSIFIER_PATH)

    regressor = xgb.XGBRegressor()
    regressor.load_model(REGRESSOR_PATH)

    print("Loading test data...")
    df = pd.read_parquet(FINAL_DIR / "final_dataset.parquet")
    test_df = df[df['year'] >= 2022].copy()

    target_col     = next(c for c in ['precipitationCal', 'precipitation'] if c in test_df.columns)
    target_log_col = f'{target_col}_log'

    print(f"Test set rows: {len(test_df):,}")

    X_test = test_df[FEATURES]
    test_df['rain_actual'] = (test_df[target_col] > 0).astype(int)

    #                                                                                                   <- Threshold sweep
    rain_proba = classifier.predict_proba(X_test)[:, 1]
    results = []

    for thresh in np.arange(0.20, 0.61, 0.01):
        preds     = (rain_proba >= thresh).astype(int)
        f1        = f1_score(test_df['rain_actual'], preds)
        recall    = (preds[test_df['rain_actual'] == 1] == 1).mean()
        precision = (test_df['rain_actual'][preds == 1] == 1).mean() if preds.sum() > 0 else 0
        results.append((thresh, f1, recall, precision))

    if RAIN_THRESHOLD is not None:
        best_thresh = RAIN_THRESHOLD
        r = next(r for r in results if round(r[0], 2) == round(best_thresh, 2))
        best_f1, best_recall, best_precision = r[1], r[2], r[3]
        print(f"\nManual threshold: {best_thresh:.2f}")
    else:
        best_thresh, best_f1, best_recall, best_precision = 0.50, 0, 0, 0
        for thresh, f1, recall, precision in results:
            if recall > best_recall and precision >= PRECISION_FLOOR:
                best_thresh, best_f1, best_recall, best_precision = thresh, f1, recall, precision
        print(f"\nAuto threshold: {best_thresh:.2f}")

    print(f"  Rain Recall:    {best_recall:.4f}")
    print(f"  Rain Precision: {best_precision:.4f}")
    print(f"  Rain F1:        {best_f1:.4f}")

    #                                                                                                  <- S1: Classify with chosen threshold 
    print(f"\nClassifying rain/no-rain (threshold={best_thresh:.2f})...")
    test_df['rain_predicted'] = (rain_proba >= best_thresh).astype(int)

    print("\nStage 1: Classifier Performance")
    print(f"Accuracy: {accuracy_score(test_df['rain_actual'], test_df['rain_predicted']):.4f}")
    print(classification_report(test_df['rain_actual'], test_df['rain_predicted'],
                                 target_names=['Dry', 'Rain']))

    #                                                                                                  <-  S2: Predict amount on predicted rainy days 
    print("Predicting precipitation amount...")
    rainy_mask = test_df['rain_predicted'] == 1
    test_df['predicted_rain'] = 0.0

    if rainy_mask.sum() > 0:
        raw_preds = regressor.predict(X_test[rainy_mask]).clip(0, LOG_CLIP)
        test_df.loc[rainy_mask, 'predicted_rain'] = np.expm1(raw_preds)

    #                                                                                                       <- Combined metrics 
    mae = mean_absolute_error(test_df[target_col], test_df['predicted_rain'])
    r2  = r2_score(test_df[target_col], test_df['predicted_rain'])

    rainy_actual = test_df[test_df[target_col] > 0]
    mae_rainy = mean_absolute_error(rainy_actual[target_col], rainy_actual['predicted_rain'])
    r2_rainy  = r2_score(rainy_actual[target_col], rainy_actual['predicted_rain'])

    print("\n Stage 2: Combined Regression Performance ")
    print(f"MAE  All days):   {mae:.4f} mm/day")
    print(f"R²   All days:   {r2:.4f}\n")
    print(f"MAE  Rainy only: {mae_rainy:.4f} mm/day")
    print(f"R²   Rainy only: {r2_rainy:.4f}")

    #                                                                                                       <- Threshold sweep plot
    thresholds = [r[0] for r in results]
    f1s        = [r[1] for r in results]
    recalls    = [r[2] for r in results]
    precisions = [r[3] for r in results]

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, f1s,        label='F1',        color='green')
    plt.plot(thresholds, recalls,    label='Recall',    color='steelblue')
    plt.plot(thresholds, precisions, label='Precision', color='orange')
    plt.axvline(best_thresh, color='red', linestyle='--', label=f'Threshold ({best_thresh:.2f})')
    plt.title("Rain Classifier — Threshold vs F1 / Recall / Precision")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("threshold_sweep.png", dpi=300)
    print("\nSaved threshold_sweep.png")

    #                                                                                                       <- Confusion matrix heatmap 
    cm = confusion_matrix(test_df['rain_actual'], test_df['rain_predicted'])
    cm_labels = np.array([
        [f"True Dry\n{cm[0,0]:,}", f"False Rain\n{cm[0,1]:,}"],
        [f"Missed Rain\n{cm[1,0]:,}", f"Caught Rain\n{cm[1,1]:,}"]
    ])

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=cm_labels, fmt='', cmap='Blues',
                xticklabels=['Predicted Dry', 'Predicted Rain'],
                yticklabels=['Actual Dry', 'Actual Rain'],
                linewidths=1, linecolor='white', cbar=False,
                annot_kws={"size": 13, "weight": "bold"})
    plt.title("Confusion Matrix — Rain Detection (2022–2024)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    print("Saved confusion_matrix.png")

    #                                                                                                       <- Seasonal pattern — actual vs predicted by year
    test_df['month'] = pd.to_datetime(test_df['time']).dt.month
    month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    colors = {2022: '#e07b39', 2023: '#6a9e5f', 2024: '#9b59b6'}

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)

    actual_monthly = test_df.groupby('month')[target_col].mean()
    axes[0].bar(range(1, 13), actual_monthly, color='steelblue', alpha=0.8)
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(month_names)
    axes[0].set_title("Actual Avg Precipitation by Month (2022–2024)")
    axes[0].set_ylabel("Precipitation (mm/day)")

    for year in [2022, 2023, 2024]:
        yr = test_df[test_df['year'] == year]
        pred_monthly = yr.groupby('month')['predicted_rain'].mean()
        axes[1].plot(pred_monthly.index, pred_monthly.values,marker='o', label=str(year), color=colors[year])

    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(month_names)
    axes[1].set_title("Predicted Avg Precipitation by Month — Per Year")
    axes[1].set_ylabel("Precipitation (mm/day)")
    axes[1].legend(title="Year")

    plt.suptitle("Seasonal Precipitation Pattern — RGV 2022–2024", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("seasonal_pattern.png", dpi=300, bbox_inches='tight')
    print("Saved seasonal_pattern.png")

    #                                                                                                                               <- Spatial precipitation map 
    spatial = test_df.groupby(['lat', 'lon']).agg(actual=(target_col,'mean'), predicted=('predicted_rain', 'mean')).reset_index()

    def to_grid(df, col):
        return df.pivot(index='lat', columns='lon', values=col)

    actual_grid    = to_grid(spatial, 'actual')
    predicted_grid = to_grid(spatial, 'predicted')

    vmin = 0
    vmax = max(spatial['actual'].max(), spatial['predicted'].max())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, grid, title in zip(
        axes,
        [actual_grid, predicted_grid],
        ['Actual Avg Precipitation (mm/day)', 'Predicted Avg Precipitation (mm/day)']
    ):
        lons = grid.columns.values
        lats = grid.index.values
        mesh = ax.pcolormesh(
            lons, lats, grid.values,
            cmap='YlOrRd', vmin=vmin, vmax=vmax,
            shading='auto'
        )
        plt.colorbar(mesh, ax=ax, label='mm/day', orientation='vertical', pad=0.02)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(-98.9, -96.6)
        ax.set_ylim(25.5, 26.7)
        ax.grid(True, linestyle='--', alpha=0.4, color='white')
        ax.set_xticks(np.arange(-98.5, -96.5, 0.5))
        ax.set_yticks(np.arange(25.5, 26.8, 0.25))
        ax.tick_params(labelsize=8)

    plt.suptitle("Spatial Precipitation Grid — RGV Test Period (2022–2024)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("spatial_map.png", dpi=300)
    print("Saved spatial_map.png")

    #                                                                                                                               <- Monthly average comparison
    test_df['month_label'] = pd.to_datetime(test_df['time']).dt.to_period('M')
    monthly = test_df.groupby('month_label')[[target_col, 'predicted_rain']].mean()

    plt.figure(figsize=(14, 5))
    plt.plot(monthly.index.astype(str), monthly[target_col], label='Actual', color='steelblue')
    plt.plot(monthly.index.astype(str), monthly['predicted_rain'], label='Predicted', color='orange', linestyle='--')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.title("Monthly Avg Precipitation — Actual vs Predicted (2022–2024)")
    plt.ylabel("Precipitation (mm/day)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("monthly_comparison.png", dpi=300)
    print("Saved monthly_comparison.png")

    print("\nAll plots saved.")

if __name__ == "__main__":
    evaluate_model()