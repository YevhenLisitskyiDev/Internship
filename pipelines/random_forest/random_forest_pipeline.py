import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.split import ExpandingWindowSplitter
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import os
import sys
import math
import json
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the models directory to path for importing
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
models_dir = os.path.join(project_root, 'models', 'random_forest')
sys.path.insert(0, models_dir)

from random_forest_model import RandomForestForecastModel

# Experiment description
environment_desc = (
    "Random Forest | Expanding-window CV | 1 week ahead | 4-week steps | Context length 12 | "
    "Lags 1,2,4,8,12 | Seasonal features | Feature selection"
)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set local data path to the dataset in the data directory
data_path = os.path.join(project_root, 'data', 'WHO-COVID-19-global-data.csv')
if not os.path.exists(data_path):
    print(f"Data file not found at {data_path}. Please place 'WHO-COVID-19-global-data.csv' in the data directory.")
    sys.exit(1)

# Load & filter to single country
df = pd.read_csv(data_path, parse_dates=['Date_reported'])
print(f"Dataset columns: {df.columns.tolist()}")
print(f"Dataset shape: {df.shape}")

country = "Italy"
series_df = (
    df[df['Country'] == country]
      .sort_values('Date_reported')
      .reset_index(drop=True)
)
series_df['New_cases'] = series_df['New_cases'].fillna(0)

# Filter to reasonable date range (COVID data should end around 2023)
series_df = series_df[
    (series_df['Date_reported'] >= pd.Timestamp('2020-01-01')) & 
    (series_df['Date_reported'] < pd.Timestamp('2024-01-01'))
]

# Keep only necessary columns to avoid issues
necessary_cols = ['Date_reported', 'Country', 'New_cases']
available_cols = [col for col in necessary_cols if col in series_df.columns]
series_df = series_df[available_cols]

print(f"Filtered data shape: {series_df.shape}")
print(f"Date range: {series_df['Date_reported'].min()} to {series_df['Date_reported'].max()}")
print(f"Sample data:")
print(series_df.head())
print(f"Sample recent data:")
print(series_df.tail())

# Check the actual date frequency
date_diffs = pd.to_datetime(series_df['Date_reported']).diff().dropna()
print(f"Date differences (days): {date_diffs.dt.days.unique()}")
print(f"Data appears to be: {'Weekly' if date_diffs.dt.days.mode()[0] == 7 else 'Daily'}")
print(f"Total data points: {len(series_df)}")
print(f"Data from {series_df['Date_reported'].min().date()} to {series_df['Date_reported'].max().date()}")

# Check dates around 2023
dates_2022_end = series_df[series_df['Date_reported'] >= pd.Timestamp('2022-11-01')]['Date_reported']
print(f"Dates from Nov 2022 onwards: {dates_2022_end.head(10).dt.date.tolist()}")
dates_2023 = series_df[series_df['Date_reported'] >= pd.Timestamp('2023-01-01')]['Date_reported']
print(f"First few 2023 dates: {dates_2023.head(5).dt.date.tolist() if len(dates_2023) > 0 else 'No 2023 dates found!'}")

# --- Hyperparameters ---
context_length = 12  # Number of historical weeks to consider
prediction_length = 1  # 1 week ahead prediction
step_length = 4  # Step size for expanding window CV (weeks)
lags = [1, 2, 4, 8, 12]  # Weekly lags
n_estimators = 200
max_depth = 10
n_features_to_select = 8  # Use feature selection

# Compute required minimum initial window size
max_lag = max(lags)
history_len = context_length + max_lag
min_initial = history_len + prediction_length

# Expanding-window splitter
cv = ExpandingWindowSplitter(
    fh=prediction_length,
    initial_window=min_initial,
    step_length=step_length
)

# Prepare for cumulative plotting
plt.figure(figsize=(14, 7))
colors = plt.cm.get_cmap('tab10')

# Run CV folds
metrics = []
all_folds = []
all_predictions = []

for fold, (train_idx, test_idx) in enumerate(cv.split(series_df), start=1):
    # End CV loop at beginning of 2023
    first_test_date = series_df['Date_reported'].iloc[test_idx[0]]
    
    # Print test date for EVERY fold so we can see what's happening
    print(f"\n▶ Fold {fold} | Test start date: {first_test_date.date()}")
    
    if first_test_date >= pd.Timestamp('2023-01-01'):
        print(f"✅ STOPPING CV at start of 2023! Reached fold {fold} with test date {first_test_date.date()}")
        break
    
    # Prepare train and test slices
    train_df = series_df.iloc[train_idx].copy()
    test_df = series_df.iloc[test_idx].copy()
    
    # Instantiate fresh model per fold
    model = RandomForestForecastModel(
        context_length=context_length,
        lags_sequence=lags,
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_features_to_select=n_features_to_select,
        date_col="Date_reported",
        target_col="New_cases",
        add_seasonal_features=True,
        random_state=42
    )
    
    # Train model
    model.train(train_df)
    
    # Make prediction
    start_date = test_df['Date_reported'].iloc[0]
    pred_df = model.predict(country, start_date, prediction_length, train_df)
    
    # Get forecasts
    forecast = np.round(pred_df['forecast'].values).astype(int)
    
    # Compute metrics on original scale
    actual = test_df['New_cases'].iloc[:prediction_length].values
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - forecast) / np.maximum(actual, 1e-8))) * 100
    
    print(f"MAE: {mae:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")
    
    # Store for cumulative plotting
    all_folds.append({
        'fold': fold,
        'start_date': start_date,
        'hist_df': series_df[series_df['Date_reported'] <= start_date],
        'test_df': test_df,
        'actual': actual,
        'forecast': forecast,
        'pred_df': pred_df
    })
    
    # Store detailed predictions
    dates_json = json.dumps([str(d) for d in pred_df['date']])
    actual_json = json.dumps(actual.tolist())
    forecast_json = json.dumps(forecast.tolist())
    
    for j in range(prediction_length):
        all_predictions.append({
            'fold': fold,
            'date': pred_df['date'].iloc[j] if j < len(pred_df['date']) else None,
            'actual': actual[j] if j < len(actual) else None,
            'forecast': forecast[j] if j < len(forecast) else None,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'experiment_description': environment_desc,
            'country': country,
            'lags_sequence': lags,
            'context_length': context_length,
            'prediction_length': prediction_length,
            'step_length': step_length,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'n_features_to_select': n_features_to_select,
            'dates_json': dates_json,
            'actuals_json': actual_json,
            'forecasts_json': forecast_json
        })
    
    # Store metrics
    metrics.append({
        'fold': fold,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'experiment_description': environment_desc,
        'actual_json': json.dumps(actual.tolist()),
        'forecast_json': json.dumps(forecast.tolist())
    })

# Create DataFrame for summary metrics
current_df = pd.DataFrame(metrics)

# Save detailed predictions to CSV
predictions_df = pd.DataFrame(all_predictions)
outputs_dir = os.path.join(project_root, 'outputs')
predictions_csv = os.path.join(outputs_dir, 'randomforest_predictions.csv')
predictions_df.to_csv(predictions_csv, index=False)
print(f"Detailed predictions saved to {predictions_csv}")

# Path for results CSV
output_csv = os.path.join(outputs_dir, 'randomforest_experiment_results.csv')

# Load existing results if any, to accumulate experiments
if os.path.exists(output_csv):
    try:
        history_df = pd.read_csv(output_csv)
        combined_df = pd.concat([history_df, current_df], ignore_index=True)
    except Exception as e:
        print(f"Warning: could not load existing CSV, starting fresh. Error: {e}")
        combined_df = current_df
else:
    combined_df = current_df

# Save the combined results back to CSV
combined_df.to_csv(output_csv, index=False)
print(f"Summary results saved/appended to {output_csv}")

# Summary statistics
if len(current_df) > 0:
    print(f"\n📊 Summary Statistics for RANDOM FOREST:")
    print(f"Average MAE:  {current_df['mae'].mean():.2f} ± {current_df['mae'].std():.2f}")
    print(f"Average RMSE: {current_df['rmse'].mean():.2f} ± {current_df['rmse'].std():.2f}")
    print(f"Average MAPE: {current_df['mape'].mean():.2f}% ± {current_df['mape'].std():.2f}%")
    print(f"Total folds completed: {len(current_df)}")

print("🎯 RANDOM FOREST pipeline execution completed!")

# Add plotting logic
if len(all_folds) > 0:
    import math
    
    # Cumulative plot after all folds
    n_folds = len(all_folds)
    n_cols = 2 if n_folds <= 4 else 3
    n_rows = math.ceil(n_folds / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=True, sharey=True)
    if n_folds == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    for i, fold_data in enumerate(all_folds):
        ax = axes[i] if n_folds > 1 else axes[0]
        
        # Plot historical data
        ax.plot(fold_data['hist_df']['Date_reported'], fold_data['hist_df']['New_cases'], 
                label='Historical', color='blue', linewidth=2)
        
        # Plot actual values for this fold
        ax.plot(fold_data['test_df']['Date_reported'].iloc[:prediction_length], fold_data['actual'], 
                label='Actual', color='green', marker='o', linestyle='-', markersize=8)
        
        # Plot forecast for this fold
        ax.plot(fold_data['pred_df']['date'], fold_data['forecast'], 
                label='Forecast', color='red', marker='x', linestyle='--', markersize=8)
        
        # Mark forecast start
        ax.axvline(fold_data['start_date'], color='black', linestyle=':', alpha=0.5)
        
        # Add metrics to title
        mae_val = metrics[i]['mae']
        rmse_val = metrics[i]['rmse']
        ax.set_title(f'Fold {fold_data["fold"]} | MAE: {mae_val:.1f} | RMSE: {rmse_val:.1f}')
        ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=45)

    # Hide any unused subplots
    if n_folds > 1:
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

    fig.suptitle(f'Random Forest 1-Week Ahead Forecasts for {country}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    print(f"📈 Displayed plots for {len(all_folds)} CV folds")
else:
    print("⚠️ No folds data available for plotting") 