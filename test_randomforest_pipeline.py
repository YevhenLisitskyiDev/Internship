import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.split import ExpandingWindowSplitter
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import os
import sys
import math
import json
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

class RandomForestForecastModel:
    """Random Forest model for time series forecasting with lag features"""
    
    def __init__(self, 
                 context_length=12,
                 lags_sequence=[1, 2, 4, 8, 12],
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 random_state=42,
                 n_features_to_select=None,
                 date_col="Date_reported",
                 target_col="New_cases",
                 add_seasonal_features=True):
        
        self.context_length = context_length
        self.lags_sequence = lags_sequence
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_features_to_select = n_features_to_select
        self.date_col = date_col
        self.target_col = target_col
        self.add_seasonal_features = add_seasonal_features
        
        # Initialize models
        self.rf_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.rfe_model = None
        self.feature_names = None
        self.max_lag = max(self.lags_sequence)
        
    def create_features(self, df):
        """Create lag features and seasonal features"""
        df_features = df.copy()
        
        # Create lag features
        for lag in self.lags_sequence:
            df_features[f'lag_{lag}'] = df_features[self.target_col].shift(lag)
        
        # Add seasonal features if requested
        if self.add_seasonal_features:
            df_features[self.date_col] = pd.to_datetime(df_features[self.date_col])
            df_features['week_of_year'] = df_features[self.date_col].dt.isocalendar().week
            df_features['month'] = df_features[self.date_col].dt.month
            df_features['quarter'] = df_features[self.date_col].dt.quarter
            df_features['year'] = df_features[self.date_col].dt.year
            
            # Cyclical encoding for week of year
            df_features['week_sin'] = np.sin(2 * np.pi * df_features['week_of_year'] / 52)
            df_features['week_cos'] = np.cos(2 * np.pi * df_features['week_of_year'] / 52)
            
            # Cyclical encoding for month
            df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
            df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        return df_features
    
    def prepare_training_data(self, df):
        """Prepare training data with features"""
        df_features = self.create_features(df)
        
        # Remove rows with NaN values (due to lagging)
        df_clean = df_features.dropna()
        
        # Define explicit numeric feature columns
        lag_cols = [f'lag_{lag}' for lag in self.lags_sequence]
        seasonal_cols = []
        
        if self.add_seasonal_features:
            seasonal_cols = [
                'week_of_year', 'month', 'quarter', 'year',
                'week_sin', 'week_cos', 'month_sin', 'month_cos'
            ]
        
        # Only include columns that actually exist and are numeric
        feature_cols = []
        for col in lag_cols + seasonal_cols:
            if col in df_clean.columns:
                # Check if the column is numeric
                try:
                    pd.to_numeric(df_clean[col], errors='raise')
                    feature_cols.append(col)
                except (ValueError, TypeError):
                    print(f"Warning: Skipping non-numeric column {col}")
        
        if not feature_cols:
            raise ValueError("No valid numeric features found!")
        
        X = df_clean[feature_cols]
        y = df_clean[self.target_col]
        
        self.feature_names = feature_cols
        
        print(f"Using features: {feature_cols}")
        
        return X, y, df_clean
    
    def train(self, train_df):
        """Train the Random Forest model"""
        print("Training Random Forest model...")
        
        X_train, y_train, _ = self.prepare_training_data(train_df)
        
        # Store original feature names for RFE
        self.original_feature_names = self.feature_names.copy()
        
        # Apply feature selection if specified
        if self.n_features_to_select and self.n_features_to_select < len(X_train.columns):
            self.rfe_model = RFE(
                estimator=RandomForestRegressor(
                    n_estimators=50,  # Smaller for RFE
                    random_state=self.random_state
                ),
                n_features_to_select=self.n_features_to_select
            )
            X_train_selected = self.rfe_model.fit_transform(X_train, y_train)
            
            # Update feature names after selection
            selected_features = np.array(self.original_feature_names)[self.rfe_model.support_]
            self.selected_feature_names = selected_features.tolist()
            
            # Train final model on selected features
            self.rf_model.fit(X_train_selected, y_train)
        else:
            # Train on all features
            self.selected_feature_names = self.feature_names.copy()
            self.rf_model.fit(X_train, y_train)
        
        print(f"Model trained with {len(self.selected_feature_names)} selected features out of {len(self.original_feature_names)} total features")
        
        # Print feature importance
        if hasattr(self.rf_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.selected_feature_names,
                'importance': self.rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nTop 10 Feature Importances:")
            print(importance_df.head(10).to_string(index=False))
    
    def predict(self, country, start_date, prediction_length, historical_df):
        """Make predictions for the specified period"""
        
        # Create a dataframe with the prediction dates
        prediction_dates = pd.date_range(
            start=start_date,
            periods=prediction_length,
            freq='W'  # Weekly frequency
        )
        
        # Get the most recent data for creating features
        recent_data = historical_df.tail(self.max_lag + 20).copy()  # Extra buffer
        
        predictions = []
        
        for i, pred_date in enumerate(prediction_dates):
            # Create features for prediction
            # We need to extend the historical data with the prediction date
            pred_row = pd.DataFrame({
                self.date_col: [pred_date],
                self.target_col: [0],  # Placeholder, will be predicted
            })
            
            # Add Country column if it exists in recent_data
            if 'Country' in recent_data.columns:
                pred_row['Country'] = [country]
            
            # Combine with recent data
            extended_data = pd.concat([recent_data, pred_row], ignore_index=True)
            
            # Create features
            df_features = self.create_features(extended_data)
            
            if i == 0:  # Debug for first prediction only
                print(f"Available features in df_features: {df_features.columns.tolist()}")
                print(f"Required original features: {self.original_feature_names}")
                print(f"Selected features: {self.selected_feature_names}")
                missing_features = [f for f in self.original_feature_names if f not in df_features.columns]
                if missing_features:
                    print(f"Missing features: {missing_features}")
            
            # Ensure we have all required ORIGINAL features (for RFE transform), filling missing ones with 0
            feature_row = pd.DataFrame(index=[0])
            for col in self.original_feature_names:
                if col in df_features.columns:
                    value = df_features.iloc[-1][col]
                    # Convert to numeric, handling any potential issues
                    try:
                        feature_row.loc[0, col] = pd.to_numeric(value, errors='coerce')
                    except:
                        feature_row.loc[0, col] = 0
                        print(f"Warning: Could not convert {col}={value} to numeric, using 0")
                else:
                    feature_row.loc[0, col] = 0
                    print(f"Warning: Feature {col} not found, using 0")
            
            # Handle any remaining missing values and ensure numeric types
            feature_row = feature_row.fillna(0).astype(float)
            
            # Apply feature selection if used during training
            if self.rfe_model is not None:
                feature_row_selected = self.rfe_model.transform(feature_row)
                prediction = self.rf_model.predict(feature_row_selected)[0]
            else:
                prediction = self.rf_model.predict(feature_row)[0]
            
            # Ensure non-negative predictions for case counts
            prediction = max(0, prediction)
            predictions.append(prediction)
            
            # Update the recent_data with the prediction for next iteration
            new_row = pd.DataFrame({
                self.date_col: [pred_date],
                self.target_col: [prediction],
            })
            if 'Country' in recent_data.columns:
                new_row['Country'] = [country]
                
            recent_data = pd.concat([recent_data, new_row], ignore_index=True)
        
        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'date': prediction_dates,
            'forecast': predictions
        })
        
        return pred_df

# Experiment description
environment_desc = (
    "Random Forest | Expanding-window CV | 1 week ahead | 4-week steps | Context length 12 | "
    "Lags 1,2,4,8,12 | Seasonal features | Feature selection"
)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Set local data path to the dataset in the root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'WHO-COVID-19-global-data.csv')
if not os.path.exists(data_path):
    print(f"Data file not found at {data_path}. Please place 'WHO-COVID-19-global-data.csv' in the script directory.")
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
lags = [1, 2, 4, 8, 12]  # Weekly lags
n_estimators = 200
max_depth = 10
n_features_to_select = 8  # Use feature selection

# Compute required minimum initial window size
max_lag = max(lags)
history_len = context_length + max_lag
min_initial = history_len + prediction_length

# Expanding-window splitter with 4-week steps
cv = ExpandingWindowSplitter(
    fh=prediction_length,
    initial_window=min_initial,
    step_length=4  # Move test window by 4 weeks each fold
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
predictions_csv = os.path.join(script_dir, 'randomforest_predictions.csv')
predictions_df.to_csv(predictions_csv, index=False)
print(f"Detailed predictions saved to {predictions_csv}")

# Path for results CSV
output_csv = os.path.join(script_dir, 'randomforest_experiment_results.csv')

# Load existing results if any
if os.path.exists(output_csv):
    try:
        history_df = pd.read_csv(output_csv)
        combined_df = pd.concat([history_df, current_df], ignore_index=True)
    except Exception as e:
        print(f"Warning: could not load existing CSV, starting fresh. Error: {e}")
        combined_df = current_df
else:
    combined_df = current_df

# Save combined results
combined_df.to_csv(output_csv, index=False)
print(f"Summary results saved/appended to {output_csv}")

# Calculate overall statistics
print(f"\n=== OVERALL RESULTS ===")
print(f"Number of folds: {len(metrics)}")
print(f"Average MAE: {np.mean([m['mae'] for m in metrics]):.2f}")
print(f"Average RMSE: {np.mean([m['rmse'] for m in metrics]):.2f}")
print(f"Average MAPE: {np.mean([m['mape'] for m in metrics]):.2f}%")

# Cumulative plot after all folds
n_folds = len(all_folds)
n_cols = 2 if n_folds <= 4 else 3
n_rows = math.ceil(n_folds / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=True, sharey=True)
if n_folds == 1:
    axes = [axes]
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

input('Press Enter to exit...') 