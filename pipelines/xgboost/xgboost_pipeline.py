import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.split import ExpandingWindowSplitter
import logging
import os
import sys
import math
import json

# Add the models directory to path for importing
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
transformer_models_dir = os.path.join(project_root, 'models', 'transformer')
xgboost_models_dir = os.path.join(project_root, 'models', 'xgboost')
sys.path.insert(0, transformer_models_dir)
sys.path.insert(0, xgboost_models_dir)

from timeseries_model import TransformerForecastModel
from xgboost_model import XGBoostForecastModel

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

# Set local data path to the dataset in the data directory
data_path = os.path.join(project_root, 'data', 'WHO-COVID-19-global-data.csv')
if not os.path.exists(data_path):
    print(f"Data file not found at {data_path}. Please place 'WHO-COVID-19-global-data.csv' in the data directory.")
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
step_length       = 4  # Step size for expanding window CV (weeks)
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
min_initial = history_len + prediction_length    # 96 + 1 = 97

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
all_folds = []  # Store data for plotting
all_predictions = []

for fold, (train_idx, test_idx) in enumerate(cv.split(series_df), start=1):
    # End CV loop at beginning of 2023 (first test date in 2023 or later)
    first_test_date = series_df['Date_reported'].iloc[test_idx[0]]
    
    print(f"\n▶ Fold {fold} | Test start date: {first_test_date.date()}")
    
    if first_test_date >= pd.Timestamp('2023-01-01'):
        print(f"✅ STOPPING CV at start of 2023! Reached fold {fold} with test date {first_test_date.date()}")
        break

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
    pred_norm_df = model.predict(country, start_date, prediction_length, strategy="recursive")

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
            'step_length': step_length,
            'model_type': MODEL_TYPE,
            'dates_json': dates_json,
            'actuals_json': actual_json,
            'forecasts_json': forecast_json
        }
        
        # Add model-specific parameters
        if MODEL_TYPE == "xgboost":
            prediction_record.update(model_params)
        else:
            prediction_record.update(model_params)
            
        all_predictions.append(prediction_record)

    # Save actual and forecast arrays as JSON strings in metrics
    metrics.append({
        'fold': fold,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'experiment_description': environment_desc,
        'model_type': MODEL_TYPE,
        'actual_json': json.dumps(actual.tolist() if isinstance(actual, np.ndarray) else list(actual)),
        'forecast_json': json.dumps(forecast.tolist() if isinstance(forecast, np.ndarray) else list(forecast))
    })

# Create DataFrame for summary metrics before using it
current_df = pd.DataFrame(metrics)

# Save detailed predictions to CSV
predictions_df = pd.DataFrame(all_predictions)
outputs_dir = os.path.join(project_root, 'outputs')
predictions_csv = os.path.join(outputs_dir, f'{MODEL_TYPE}_predictions.csv')
predictions_df.to_csv(predictions_csv, index=False)
print(f"Detailed predictions saved to {predictions_csv}")

# Path for results CSV (define before use)
output_csv = os.path.join(outputs_dir, f'{MODEL_TYPE}_experiment_results.csv')

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

# Summary statistics
if len(current_df) > 0:
    print(f"\n📊 Summary Statistics for {MODEL_TYPE.upper()}:")
    print(f"Average MAE:  {current_df['mae'].mean():.2f} ± {current_df['mae'].std():.2f}")
    print(f"Average RMSE: {current_df['rmse'].mean():.2f} ± {current_df['rmse'].std():.2f}")
    print(f"Average MAPE: {current_df['mape'].mean():.2f}% ± {current_df['mape'].std():.2f}%")
    print(f"Total folds completed: {len(current_df)}")

print(f"🎯 {MODEL_TYPE.upper()} pipeline execution completed!")

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
        ax.plot(fold_data['pred_norm_df']['date'], fold_data['forecast'], 
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

    fig.suptitle(f'{MODEL_TYPE.upper()} 1-Week Ahead Forecasts for {country}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    print(f"📈 Displayed plots for {len(all_folds)} CV folds")
else:
    print("⚠️ No folds data available for plotting") 