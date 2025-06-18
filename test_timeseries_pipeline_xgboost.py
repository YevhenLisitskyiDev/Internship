import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.split import ExpandingWindowSplitter
import logging
import os
import sys
import math
import json

# Dynamically add the time_series_transformer directory (where timeseries_model.py is) to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
timeseries_dir = os.path.join(script_dir, 'time_series_transformer')
if timeseries_dir not in sys.path:
    sys.path.insert(0, timeseries_dir)

try:
    from timeseries_model import TransformerForecastModel
    from xgboost_model import XGBoostForecastModel
except ImportError as e:
    print("Could not import models. Please check your directory structure.")
    raise e

# Model selection - Switch between 'transformer' and 'xgboost'
MODEL_TYPE = "xgboost"  # Change this to "transformer" to use the original model

# Experiment description
if MODEL_TYPE == "xgboost":
    environment_desc = (
        "Expanding-window CV XGBoost | Log transform | Context length 48 | Lags 1, 4, 8, 12, 24, 48 | Multi-step forecasting"
    )
else:
    environment_desc = (
        "Expanding-window CV No preprocessing | 20 epochs | Log transform | Context length 48 | Lags 1, 4, 8, 12, 24, 48 | 90% percentile"
    )

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set local data path to the dataset in the root directory
data_path = os.path.join(script_dir, 'WHO-COVID-19-global-data.csv')
if not os.path.exists(data_path):
    print(f"Data file not found at {data_path}. Please place 'WHO-COVID-19-global-data.csv' in the script directory.")
    sys.exit(1)

# Load & filter to single country
df = pd.read_csv(data_path, parse_dates=['Date_reported'])
country = "Italy"
series_df = (
    df[df['Country'] == country]
      .sort_values('Date_reported')
      .reset_index(drop=True)
)
series_df['New_cases'] = series_df['New_cases'].fillna(0)

# --- Hyperparameters ---
context_length    = 48
prediction_length = 1  # One week prediction (data is weekly)
lags              = [1, 4, 8, 12, 24, 48]

# Model-specific parameters
if MODEL_TYPE == "xgboost":
    # XGBoost parameters
    model_params = {
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 100,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }
else:
    # Transformer parameters
    model_params = {
        'epochs': 20,
        'learning_rate': 1e-4,
        'weight_decay': 0.01
    }

# Compute required minimum initial window size
max_lag     = max(lags)                          # e.g., 48
history_len = context_length + max_lag           # 48 + 48 = 96
min_initial = history_len + prediction_length    # 96 + 4  = 100

# Expanding-window splitter
cv = ExpandingWindowSplitter(
    fh=prediction_length,
    initial_window=min_initial,
    step_length=prediction_length
)

# Prepare for cumulative plotting
plt.figure(figsize=(14, 7))
colors = plt.cm.get_cmap('tab10')

# Run CV folds
metrics = []
all_folds = []  # Store data for plotting
all_predictions = []

for fold, (train_idx, test_idx) in enumerate(cv.split(series_df), start=1):
    # End CV loop at beginning of 2023 (first test date in 2023 or later)
    first_test_date = series_df['Date_reported'].iloc[test_idx[0]]
    if first_test_date >= pd.Timestamp('2023-01-01'):
        print(f"Stopping CV at start of 2023, reached fold {fold} with test date {first_test_date.date()}")
        break
    print(f"\n▶ Fold {fold}")

    # Prepare train and test slices
    train_df = series_df.iloc[train_idx].copy()
    test_df  = series_df.iloc[test_idx].copy()

    # Instantiate model based on selection
    if MODEL_TYPE == "xgboost":
        model = XGBoostForecastModel(
            countries=[country],
            context_length=context_length,
            prediction_length=prediction_length,
            lags_sequence=lags,
            date_col="Date_reported",
            country_col="Country",
            target_col="New_cases",
            log_transform=True,
            include_datetime_features=True,
            include_rolling_features=True,
            rolling_windows=[3, 7, 14, 30],
            **model_params
        )
    else:
        model = TransformerForecastModel(
            countries=[country],
            context_length=context_length,
            prediction_length=prediction_length,
            lags_sequence=lags,
            date_col="Date_reported",
            country_col="Country",
            target_col="New_cases",
            distribution_output='student_t',
            embedding_dim=32,
            num_parallel_samples=100,
            log_transform=True,
            device='cpu',
            **model_params
        )

    # Train & forecast on raw data
    training_results = model.train(train_df)
    start_date = test_df['Date_reported'].iloc[0]
    pred_norm_df = model.predict(country, start_date, prediction_length, strategy="direct")

    # Process forecast results
    forecast = np.round(pred_norm_df['forecast'].values).astype(int)

    # Compute full metrics on original scale
    actual   = test_df['New_cases'].iloc[:prediction_length].values
    mae  = np.mean(np.abs(actual - forecast))
    mse  = np.mean((actual - forecast) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - forecast) / np.maximum(actual, 1e-8))) * 100

    print(f"MAE:  {mae:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")

    # Print feature importance for XGBoost
    if MODEL_TYPE == "xgboost" and hasattr(model, 'get_feature_importance'):
        try:
            importance = model.get_feature_importance()
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
            print("Top 5 features:")
            for feature, importance_score in top_features:
                print(f"  {feature}: {importance_score:.4f}")
        except Exception as e:
            print(f"Could not get feature importance: {e}")

    # Store for cumulative plotting
    all_folds.append({
        'fold': fold,
        'start_date': start_date,
        'hist_df': series_df[series_df['Date_reported'] <= start_date],
        'test_df': test_df,
        'actual': actual,
        'forecast': forecast,
        'pred_norm_df': pred_norm_df
    })

    # Store detailed predictions for this fold
    # Prepare JSON strings for the full arrays for this fold
    dates_json = json.dumps([str(d) for d in pred_norm_df['date']])
    actual_json = json.dumps(actual.tolist() if isinstance(actual, np.ndarray) else list(actual))
    forecast_json = json.dumps(forecast.tolist() if isinstance(forecast, np.ndarray) else list(forecast))

    for j in range(prediction_length):
        prediction_record = {
            'fold': fold,
            'date': pred_norm_df['date'].iloc[j] if j < len(pred_norm_df['date']) else None,
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
            'model_type': MODEL_TYPE,
            'dates_json': dates_json,
            'actuals_json': actual_json,
            'forecasts_json': forecast_json
        }
        
        # Add model-specific parameters
        if MODEL_TYPE == "xgboost":
            prediction_record.update({
                'learning_rate': model_params['learning_rate'],
                'max_depth': model_params['max_depth'],
                'n_estimators': model_params['n_estimators']
            })
        else:
            prediction_record.update({
                'epochs': model_params['epochs'],
                'learning_rate': model_params['learning_rate'],
                'weight_decay': model_params['weight_decay']
            })
            
        all_predictions.append(prediction_record)

    # Save actual and forecast arrays as JSON strings in metrics
    metrics_record = {
        'fold': fold,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'experiment_description': environment_desc,
        'model_type': MODEL_TYPE,
        'actual_json': json.dumps(actual.tolist() if isinstance(actual, np.ndarray) else list(actual)),
        'forecast_json': json.dumps(forecast.tolist() if isinstance(forecast, np.ndarray) else list(forecast))
    }
    metrics.append(metrics_record)

# Create DataFrame for summary metrics before using it
current_df = pd.DataFrame(metrics)

# Save detailed predictions to CSV
predictions_df = pd.DataFrame(all_predictions)
predictions_csv = os.path.join(script_dir, f'expanding_window_predictions_{MODEL_TYPE}.csv')
predictions_df.to_csv(predictions_csv, index=False)
print(f"Detailed predictions saved to {predictions_csv}")

# Path for results CSV (define before use)
output_csv = os.path.join(script_dir, f'expanding_window_experiment_results_{MODEL_TYPE}.csv')

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

# Save the combined results back to CSV (append mode, no deduplication)
combined_df.to_csv(output_csv, index=False)
print(f"Summary results saved/appended to {output_csv}")

# Use combined DataFrame for further analysis
metrics_df = combined_df

# Cumulative plot after all folds (grid of subplots)
n_folds = len(all_folds)
n_cols = 2 if n_folds <= 4 else 3
n_rows = math.ceil(n_folds / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=True, sharey=True)
axes = axes.flatten()  # To index easily

for i, fold_data in enumerate(all_folds):
    ax = axes[i]
    
    # Show only last 60 days of historical data for better detail
    start_date = fold_data['start_date']
    hist_window_start = start_date - pd.Timedelta(days=60)
    hist_window_data = fold_data['hist_df'][fold_data['hist_df']['Date_reported'] >= hist_window_start]
    
    # Plot historical data (last 60 days)
    ax.plot(hist_window_data['Date_reported'], hist_window_data['New_cases'], 
            label='Historical', color='blue', linewidth=2, alpha=0.7)
    
    # Get the actual prediction dates from the model output
    pred_dates = fold_data['pred_norm_df']['date'].values
    actual_values = fold_data['actual']
    forecast_values = fold_data['forecast']
    
    # Ensure all arrays have the same length (take minimum to avoid index errors)
    min_length = min(len(pred_dates), len(actual_values), len(forecast_values))
    pred_dates = pred_dates[:min_length]
    actual_values = actual_values[:min_length]
    forecast_values = forecast_values[:min_length]
    
    # Ensure dates are pandas Timestamp objects for proper plotting
    if len(pred_dates) > 0 and not isinstance(pred_dates[0], pd.Timestamp):
        pred_dates = pd.to_datetime(pred_dates)
    
    # Plot actual values for the prediction period with correct dates
    if len(pred_dates) > 0 and len(actual_values) > 0:
        ax.plot(pred_dates, actual_values, 
                label='Actual', color='green', marker='o', markersize=10, 
                linewidth=3, linestyle='-', markeredgecolor='black', markeredgewidth=1)
    
    # Plot forecast with different colors/markers for each step to show multi-step clearly
    forecast_colors = ['red', 'orange', 'purple', 'brown']
    forecast_markers = ['X', 's', '^', 'D']
    
    # Determine if data is weekly or daily based on actual historical data frequency
    if len(fold_data['hist_df']) >= 2:
        # Check frequency of historical data
        hist_dates = pd.to_datetime(fold_data['hist_df']['Date_reported']).sort_values()
        hist_freq_days = (hist_dates.iloc[-1] - hist_dates.iloc[-2]).days
        date_freq = "Week" if hist_freq_days >= 7 else "Day"
    else:
        date_freq = "Week"  # Default assumption for COVID data
    
    # Debug: print prediction info for first few folds
    if i < 3:
        print(f"Fold {fold_data['fold']}: pred_dates length={len(pred_dates)}, actual length={len(actual_values)}, forecast length={len(forecast_values)}")
        print(f"  Date freq detected: {date_freq}")
        if len(pred_dates) >= 2:
            print(f"  Days between predictions: {(pred_dates[1] - pred_dates[0]).days}")
    
    for step in range(len(pred_dates)):
        if step >= len(forecast_values):  # Safety check
            break
            
        step_date = pred_dates[step]
        step_forecast = forecast_values[step]
        color = forecast_colors[step % len(forecast_colors)]
        marker = forecast_markers[step % len(forecast_markers)]
        
        ax.plot(step_date, step_forecast, 
                color=color, marker=marker, markersize=10, 
                label=f'Pred {date_freq} {step+1}' if i == 0 else "",  # Only show legend for first plot
                markeredgecolor='black', markeredgewidth=1)
    
    # Connect forecast points with a line for clarity
    if len(pred_dates) > 1 and len(forecast_values) > 1:
        ax.plot(pred_dates, forecast_values, 
                color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Mark forecast start with a more visible line
    ax.axvline(start_date, color='black', linestyle=':', alpha=0.8, linewidth=2)
    
    # Add text annotation showing the prediction period with MAE
    if len(pred_dates) > 0 and len(actual_values) > 0 and len(forecast_values) > 0:
        pred_start = pred_dates[0]
        pred_end = pred_dates[-1]
        mae_for_fold = np.mean(np.abs(actual_values - forecast_values))
        ax.text(0.02, 0.98, f'Pred: {pd.to_datetime(pred_start).strftime("%m/%d")} - {pd.to_datetime(pred_end).strftime("%m/%d")}\nMAE: {mae_for_fold:.0f}', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title(f'Fold {fold_data["fold"]} ({MODEL_TYPE.upper()})')
    
    # Improve legend to show all elements clearly
    if i == 0:  # Only show full legend on first plot
        ax.legend(fontsize=8, loc='upper right')
    else:
        # Show minimal legend on other plots
        handles, labels = ax.get_legend_handles_labels()
        # Only show Historical and Actual in legend for clarity
        hist_actual_handles = [h for h, l in zip(handles, labels) if l in ['Historical', 'Actual']]
        hist_actual_labels = [l for l in labels if l in ['Historical', 'Actual']]
        if hist_actual_handles:
            ax.legend(hist_actual_handles, hist_actual_labels, fontsize=8, loc='upper right')
    
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to better show the data range
    all_values = list(actual_values) + list(forecast_values)
    if hist_window_data is not None and len(hist_window_data) > 0:
        all_values.extend(hist_window_data['New_cases'].tolist())
    
    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        y_padding = (y_max - y_min) * 0.1
        ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)

# Hide any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle(f'Expanding Window Forecasts for {country} - {MODEL_TYPE.upper()}', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

print(f"\nCompleted {MODEL_TYPE.upper()} pipeline experiment!")
print(f"Average metrics across all folds:")
print(f"MAE: {current_df['mae'].mean():.2f} ± {current_df['mae'].std():.2f}")
print(f"RMSE: {current_df['rmse'].mean():.2f} ± {current_df['rmse'].std():.2f}")
print(f"MAPE: {current_df['mape'].mean():.2f}% ± {current_df['mape'].std():.2f}%")

input('Press Enter to exit...') 