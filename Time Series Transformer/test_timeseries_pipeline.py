import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.split import ExpandingWindowSplitter
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
import os
import warnings
warnings.filterwarnings('ignore')
import torch # Added import for torch, as it's used in model instantiation
from timeseries_model import TransformerForecastModel

plt.ion()

print("DEBUG: Initial imports successful")

# Experiment description
experiment_desc = (
    "Expanding-window CV with adaptive scaling: "
    "RobustScaler for high variance, StandardScaler for normal data, "
    "enhanced features, seasonal lags"
)
print(f"DEBUG: Experiment description set: {experiment_desc}")

# Update data path to local dataset in project root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(root_dir, 'WHO-COVID-19-global-data.csv')
print(f"DEBUG: Data path: {data_path}")
try:
    df = pd.read_csv(data_path, parse_dates=['Date_reported'])
    print(f"DEBUG: Loaded data from path. DataFrame shape: {df.shape}")
    if not df.empty:
        print(f"DEBUG: df.head():\n{df.head()}")
except Exception as e_load:
    print(f"DEBUG: ERROR loading data from {data_path}: {e_load}")
    raise

country = 'Republic of Korea'
print(f"DEBUG: Selected country: {country}")
series_df = (
    df[df['Country'] == country]
      .sort_values('Date_reported')
      .reset_index(drop=True)
)
print(f"DEBUG: Filtered DataFrame for {country}. series_df shape: {series_df.shape}")
if not series_df.empty:
    print(f"DEBUG: series_df.head():\n{series_df.head()}")
else:
    print("DEBUG: series_df is empty after filtering!")

# Advanced preprocessing
def preprocess_covid_data(input_df, target_col='New_cases'): # Renamed df to input_df for clarity in debugs
    """Apply advanced preprocessing for COVID data"""
    print(f"DEBUG: [preprocess_covid_data] Called with df shape: {input_df.shape}, target_col: {target_col}")
    # df = df.copy() # Original line
    processed_df = input_df.copy() # Use processed_df for modifications
    print(f"DEBUG: [preprocess_covid_data] Copied input_df to processed_df.")

    # Handle missing values intelligently
    print(f"DEBUG: [preprocess_covid_data] Missing values in '{target_col}' before ffill: {processed_df[target_col].isna().sum()}")
    processed_df[target_col] = processed_df[target_col].fillna(method='ffill', limit=2)
    print(f"DEBUG: [preprocess_covid_data] Missing values after ffill (limit 2): {processed_df[target_col].isna().sum()}")
    if processed_df[target_col].isna().any():
        print(f"DEBUG: [preprocess_covid_data] Missing values before interpolate: {processed_df[target_col].isna().sum()}")
        processed_df[target_col] = processed_df[target_col].interpolate(method='linear', limit_area='inside')
        print(f"DEBUG: [preprocess_covid_data] Missing values after interpolate: {processed_df[target_col].isna().sum()}")
    processed_df[target_col] = processed_df[target_col].fillna(0)
    print(f"DEBUG: [preprocess_covid_data] Missing values after fillna(0): {processed_df[target_col].isna().sum()}")

    # Detect and handle outliers using rolling statistics
    window_param = 8 # Renamed 'window' from original to avoid conflict if 'window' is a global
    print(f"DEBUG: [preprocess_covid_data] Rolling window_param for outlier detection: {window_param}")
    rolling_median = processed_df[target_col].rolling(window=window_param, center=True, min_periods=1).median()
    rolling_std = processed_df[target_col].rolling(window=window_param, center=True, min_periods=1).std()

    epsilon_std_calc = 1e-8 # Renamed for clarity
    z_scores = np.abs((processed_df[target_col] - rolling_median) / (rolling_std + epsilon_std_calc))
    if not z_scores.empty:
        print(f"DEBUG: [preprocess_covid_data] Calculated z_scores. Example z_scores head (first 5 non-NaN):\n{z_scores.dropna().head()}")
    else:
        print(f"DEBUG: [preprocess_covid_data] z_scores Series is empty.")


    # Cap extreme outliers at 99th percentile within rolling window
    outlier_mask = z_scores > 4
    print(f"DEBUG: [preprocess_covid_data] Number of potential outliers (z_scores > 4): {outlier_mask.sum()}")

    # Iterate over indices where outlier_mask is True
    # for idx in df[outlier_mask].index: # Original line
    for idx in processed_df[outlier_mask].index: # Use processed_df
        window_start = max(0, idx - window_param // 2)
        window_end = min(len(processed_df), idx + window_param // 2 + 1) # Use processed_df for len

        # window_values = df.iloc[window_start:window_end][target_col].dropna() # Original line
        current_window_values = processed_df.iloc[window_start:window_end][target_col].dropna() # Use processed_df

        if not current_window_values.empty:
            cap_value = np.percentile(current_window_values, 99)
            original_value_at_idx = processed_df.loc[idx, target_col] # Use processed_df

            # df.loc[idx, target_col] = min(df.loc[idx, target_col], cap_value) # Original line
            new_val = min(original_value_at_idx, cap_value)
            if new_val != original_value_at_idx:
                processed_df.loc[idx, target_col] = new_val # Use processed_df
                print(f"DEBUG: [preprocess_covid_data] Capped outlier at index {idx}: Original={original_value_at_idx}, New={new_val} (Cap Value={cap_value})")
        else:
            print(f"DEBUG: [preprocess_covid_data] Window values empty for index {idx}, not capping.")

    print(f"DEBUG: [preprocess_covid_data] Finished. Returning df shape: {processed_df.shape}")
    if not processed_df.empty:
        print(f"DEBUG: [preprocess_covid_data] processed_df['{target_col}'] after preprocessing head:\n{processed_df[target_col].head()}")
    return processed_df # Return the modified copy

# Apply preprocessing
print("DEBUG: Applying preprocessing to series_df...")
series_df = preprocess_covid_data(series_df) # series_df is reassigned here
print(f"DEBUG: Preprocessing applied. series_df shape after reassignment: {series_df.shape}")
if not series_df.empty:
    print(f"DEBUG: series_df.head() after preprocessing:\n{series_df.head()}")

# Create time features function
def create_time_features(dates_arg): # Renamed 'dates' to 'dates_arg'
    """Create rich temporal features for the model"""
    print(f"DEBUG: [create_time_features] Called with dates_arg type: {type(dates_arg)}")
    if hasattr(dates_arg, '__len__') and len(dates_arg) > 0:
        print(f"DEBUG: [create_time_features] First 5 dates_arg: {dates_arg[:5]}")
    else:
        print(f"DEBUG: [create_time_features] dates_arg is empty or has no length.")

    # dates = pd.DatetimeIndex(dates) # Original line
    dates_dt_idx = pd.DatetimeIndex(dates_arg) # Use dates_arg

    # Basic temporal features
    # Ensure float division and handle potential empty arrays if dates_dt_idx is empty
    week_of_year = dates_dt_idx.isocalendar().week.to_numpy(dtype=float) / 52.0 if not dates_dt_idx.empty else np.array([])
    month = dates_dt_idx.month.to_numpy(dtype=float) / 12.0 if not dates_dt_idx.empty else np.array([])

    # Cyclical encoding for seasonality
    month_sin = np.sin(2 * np.pi * dates_dt_idx.month.to_numpy(dtype=float) / 12.0) if not dates_dt_idx.empty else np.array([])
    month_cos = np.cos(2 * np.pi * dates_dt_idx.month.to_numpy(dtype=float) / 12.0) if not dates_dt_idx.empty else np.array([])
    week_sin = np.sin(2 * np.pi * dates_dt_idx.isocalendar().week.to_numpy(dtype=float) / 52.0) if not dates_dt_idx.empty else np.array([])
    week_cos = np.cos(2 * np.pi * dates_dt_idx.isocalendar().week.to_numpy(dtype=float) / 52.0) if not dates_dt_idx.empty else np.array([])

    # COVID-specific features
    is_winter = ((dates_dt_idx.month >= 11) | (dates_dt_idx.month <= 2)).astype(float) if not dates_dt_idx.empty else np.array([])
    year_progress = dates_dt_idx.dayofyear.to_numpy(dtype=float) / 365.25 if not dates_dt_idx.empty else np.array([])

    if dates_dt_idx.empty:
        print("DEBUG: [create_time_features] Input dates_arg were empty, returning empty array for features.")
        # The number of columns should match num_time_features (8)
        return np.empty((0, 8)) # num_time_features is 8

    stacked_features = np.stack([
        week_of_year, month, month_sin, month_cos,
        week_sin, week_cos, is_winter, year_progress
    ], axis=1)
    print(f"DEBUG: [create_time_features] Shape of stacked features: {stacked_features.shape}")
    if stacked_features.shape[0] > 0:
        print(f"DEBUG: [create_time_features] Example of first row of features: {stacked_features[0]}")
    return stacked_features

# Function to choose appropriate scaler
def choose_scaler(train_values_arg, verbose=True): # Renamed train_vals to train_values_arg
    """Choose the best scaler based on data characteristics"""
    print(f"DEBUG: [choose_scaler] Called with train_values_arg length: {len(train_values_arg)}")
    if len(train_values_arg) > 0:
        print(f"DEBUG: [choose_scaler] First 5 train_values_arg: {train_values_arg[:5]}")
    else:
        print("DEBUG: [choose_scaler] train_values_arg is empty.")
        print("DEBUG: [choose_scaler]   → Defaulting to StandardScaler due to empty train_values_arg.")
        return StandardScaler()

    # Calculate statistics
    mean_val = np.mean(train_values_arg)
    std_val = np.std(train_values_arg)
    cv = std_val / (mean_val + 1e-8)

    train_series_for_skew = pd.Series(train_values_arg) # Renamed
    skewness = train_series_for_skew.skew()
    kurtosis = train_series_for_skew.kurtosis()

    # Count zeros and small values
    zero_ratio = np.sum(train_values_arg == 0) / len(train_values_arg)
    small_ratio = np.sum(train_values_arg < 10) / len(train_values_arg)

    # Original prints are modified to include "DEBUG:" prefix
    if verbose:
        print(f"DEBUG: [choose_scaler]   Data stats: CV={cv:.2f}, Skew={skewness:.2f}, Kurt={kurtosis:.2f}") # Original print with DEBUG
        print(f"DEBUG: [choose_scaler]   Zero ratio={zero_ratio:.2%}, Small value ratio={small_ratio:.2%}") # Original print with DEBUG

    # Decision logic
    chosen_scaler_type = None # Renamed
    if cv > 2.0 or kurtosis > 10:
        if verbose:
            print(f"DEBUG: [choose_scaler]   → Using RobustScaler (high variance/outliers)") # Original print with DEBUG
        chosen_scaler_type = RobustScaler(quantile_range=(10, 90))

    elif skewness > 2.0 and zero_ratio < 0.1:
        if verbose:
            print(f"DEBUG: [choose_scaler]   → Using log transform (high skewness, few zeros)") # Original print with DEBUG
        chosen_scaler_type = 'log'

    elif skewness > 1.0:
        if verbose:
            print(f"DEBUG: [choose_scaler]   → Using PowerTransformer(moderate skewness)") # Original print with DEBUG
        chosen_scaler_type = PowerTransformer(method='yeo-johnson', standardize=True)

    else:
        if verbose:
            print(f"DEBUG: [choose_scaler]   → Using StandardScaler (relatively normal distribution)") # Original print with DEBUG
        chosen_scaler_type = StandardScaler()

    print(f"DEBUG: [choose_scaler] Chosen scaler type: {type(chosen_scaler_type).__name__ if chosen_scaler_type != 'log' else 'log'}")
    return chosen_scaler_type


# Enhanced hyperparameters
context_length    = 52
prediction_length = 4
lags              = [1, 2, 3, 4, 8, 12, 24, 52]
epochs            = 20
learning_rate     = 5e-5
weight_decay      = 0.05
num_time_features = 8

print(f"DEBUG: Hyperparameters set:"
      f"\n  context_length: {context_length}"
      f"\n  prediction_length: {prediction_length}"
      f"\n  lags: {lags}"
      f"\n  epochs: {epochs}"
      f"\n  learning_rate: {learning_rate}"
      f"\n  weight_decay: {weight_decay}"
      f"\n  num_time_features: {num_time_features}")

# Compute required minimum initial window size
max_lag_val_hyper = max(lags) if lags else 0 # Renamed max_lag
history_len_hyper = context_length + max_lag_val_hyper # Renamed
min_initial_window = history_len_hyper + prediction_length # Renamed

print(f"DEBUG: max_lag_val_hyper: {max_lag_val_hyper}, history_len_hyper: {history_len_hyper}")
# Original print:
print(f"Minimum initial window size: {min_initial_window} weeks")
print(f"Total data available: {len(series_df)} weeks")

# Expanding-window splitter
# cv = ExpandingWindowSplitter(...) # Original line
cv_splitter_obj = ExpandingWindowSplitter( # Renamed cv
    # Changed fh from integer to np.arange to ensure full prediction_length in test set
    fh=np.arange(1, prediction_length + 1),
    initial_window=min_initial_window,
    step_length=prediction_length
)
print(f"DEBUG: ExpandingWindowSplitter initialized with: "
      f"fh={np.arange(1, prediction_length + 1)}, initial_window={min_initial_window}, step_length={prediction_length}")
print("DEBUG: NOTE - 'fh' is now explicitly set as np.arange(1, prediction_length + 1) to ensure "
      "the test set contains all steps from 1 to prediction_length.")


# Run CV folds
metrics_results = [] # Renamed metrics
fold_predictions_data = []  # Renamed fold_predictions
print("DEBUG: Starting CV folds...")

# Check if series_df is long enough for even one split
if len(series_df) < min_initial_window: # A split needs at least initial_window for train
    print(f"DEBUG: ERROR - series_df (len {len(series_df)}) is too short for the first CV training window "
          f"(min_initial_window: {min_initial_window}). No folds will run.")
elif len(series_df) < min_initial_window + prediction_length: # And enough data for at least one test set of prediction_length
     print(f"DEBUG: WARNING - series_df (len {len(series_df)}) is short. First train window is {min_initial_window}. "
           f"Available for test: {len(series_df) - min_initial_window}, need {prediction_length}. May skip folds.")
# else: # Sufficient data for at least one full split and test
    # for fold, (train_idx, test_idx) in enumerate(cv.split(series_df), start=1): # Original line
for fold_count, (train_indices_cv, test_indices_cv) in enumerate(cv_splitter_obj.split(series_df), start=1): # Renamed variables
    print(f"DEBUG: ----- FOLD {fold_count} -----")
    print(f"DEBUG: train_indices_cv (type {type(train_indices_cv)}): {train_indices_cv[0]}...{train_indices_cv[-1]} (length: {len(train_indices_cv)})")
    print(f"DEBUG: test_indices_cv (type {type(test_indices_cv)}): {test_indices_cv[0]}...{test_indices_cv[-1]} (length: {len(test_indices_cv)})")

    # End CV loop at beginning of 2023
    # first_test_date = series_df['Date_reported'].iloc[test_idx[0]] # Original line
    current_first_test_date = series_df['Date_reported'].iloc[test_indices_cv[0]] # Renamed
    print(f"DEBUG: Fold {fold_count}, current_first_test_date: {current_first_test_date}")
    if current_first_test_date >= pd.Timestamp('2023-01-01'):
        # print(f"\nStopping CV at start of 2023 (fold {fold}, test date: {first_test_date.date()})") # Original print
        print(f"\nStopping CV at start of 2023 (fold {fold_count}, test date: {current_first_test_date.date()})")
        print(f"DEBUG: Stopping CV loop because current_first_test_date {current_first_test_date} >= 2023-01-01")
        break

    # Skip if insufficient test data
    # if len(test_idx) < prediction_length: # Original line
    if len(test_indices_cv) < prediction_length: # Using renamed variable
        # print(f"\nSkipping fold {fold}: insufficient test data") # Original print
        print(f"\nSkipping fold {fold_count}: insufficient test data (test_indices_cv length: {len(test_indices_cv)}, prediction_length: {prediction_length})")
        print(f"DEBUG: Skipping fold {fold_count} due to insufficient test data based on original check.")
        continue

    # Original prints:
    # print(f"\n{'='*60}")
    # print(f"FOLD {fold}")
    # print(f"{'='*60}")
    print(f"\n{'='*60}")
    print(f"FOLD {fold_count}")
    print(f"{'='*60}")

    # Prepare train and test slices
    # train_df = series_df.iloc[train_idx].copy() # Original line
    # test_df  = series_df.iloc[test_idx].copy() # Original line
    current_train_df = series_df.iloc[train_indices_cv].copy() # Renamed
    current_test_df  = series_df.iloc[test_indices_cv].copy()  # Renamed
    print(f"DEBUG: current_train_df shape: {current_train_df.shape}, current_test_df shape: {current_test_df.shape}")

    # Original prints:
    # print(f"Train period: {train_df['Date_reported'].iloc[0].date()} to {train_df['Date_reported'].iloc[-1].date()}")
    # print(f"Test period:  {test_df['Date_reported'].iloc[0].date()} to {test_df['Date_reported'].iloc[prediction_length-1].date()}")
    print(f"Train period: {current_train_df['Date_reported'].iloc[0].date()} to {current_train_df['Date_reported'].iloc[-1].date()}")

    print_test_period_end_idx = min(prediction_length - 1, len(current_test_df) - 1)
    if print_test_period_end_idx < 0:
         print(f"Test period:  {current_test_df['Date_reported'].iloc[0].date()} to N/A (current_test_df too short for prediction_length print)")
    else:
         print(f"Test period:  {current_test_df['Date_reported'].iloc[0].date()} to {current_test_df['Date_reported'].iloc[print_test_period_end_idx].date()}")


    # Choose and apply appropriate scaler
    # train_vals = train_df['New_cases'].values # Original line
    fold_train_values = current_train_df['New_cases'].values # Renamed
    print(f"DEBUG: Calling choose_scaler for fold {fold_count} with fold_train_values of length {len(fold_train_values)}")
    # scaler = choose_scaler(train_vals) # Original line
    active_scaler = choose_scaler(fold_train_values, verbose=True) # Renamed scaler
    print(f"DEBUG: Chosen active_scaler for fold {fold_count}: {type(active_scaler).__name__ if active_scaler != 'log' else 'log'}")

    # Apply scaling and define inverse_transform_fold_fn
    inverse_transform_fold_fn = None # Initialize
    # if scaler == 'log': # Original line
    if active_scaler == 'log':
        epsilon_log_param = 1.0 # Renamed epsilon
        print(f"DEBUG: Applying log transform with epsilon_log_param={epsilon_log_param}")
        # train_df['New_cases_scaled'] = np.log1p(train_vals + epsilon) # Original line
        current_train_df['New_cases_scaled'] = np.log1p(fold_train_values + epsilon_log_param)
        if not current_train_df.empty:
            print(f"DEBUG: current_train_df['New_cases_scaled'] head after log transform:\n{current_train_df['New_cases_scaled'].head()}")

        # def inverse_transform(x): # Original line
        #     return np.maximum(0, np.expm1(x) - epsilon) # Original line
        def inverse_transform_log_local(x_input): # Renamed
            res_log = np.maximum(0, np.expm1(x_input) - epsilon_log_param)
            return res_log
        inverse_transform_fold_fn = inverse_transform_log_local
    else: # Sklearn scaler
        print(f"DEBUG: Applying sklearn scaler: {type(active_scaler).__name__}")
        # train_df['New_cases_scaled'] = scaler.fit_transform(train_df[['New_cases']]) # Original line
        current_train_df['New_cases_scaled'] = active_scaler.fit_transform(current_train_df[['New_cases']])
        if not current_train_df.empty:
            print(f"DEBUG: current_train_df['New_cases_scaled'] head after sklearn scaler:\n{current_train_df['New_cases_scaled'].head()}")

        # def inverse_transform(x): # Original line
        #     return scaler.inverse_transform(x.reshape(-1, 1)).flatten() # Original line
        def inverse_transform_sklearn_local(x_input): # Renamed
            x_input_reshaped = x_input.reshape(-1, 1) if x_input.ndim == 1 else x_input
            res_sklearn = active_scaler.inverse_transform(x_input_reshaped).flatten()
            return res_sklearn
        inverse_transform_fold_fn = inverse_transform_sklearn_local

    # Calculate recent statistics for post-processing
    # recent_data = train_df.tail(12) # Original line
    fold_recent_data = current_train_df.tail(12) # Renamed
    fold_recent_mean = fold_recent_data['New_cases'].mean() if not fold_recent_data.empty else np.nan
    fold_recent_max = fold_recent_data['New_cases'].max() if not fold_recent_data.empty else np.nan
    fold_recent_trend = (fold_recent_data['New_cases'].iloc[-1] - fold_recent_data['New_cases'].iloc[0]) / len(fold_recent_data) if len(fold_recent_data) > 1 else 0.0

    # Original print:
    # print(f"   Recent stats: mean={recent_mean:.0f}, max={recent_max:.0f}, trend={recent_trend:.1f}/week")
    print(f"   Recent stats: mean={fold_recent_mean:.0f}, max={fold_recent_max:.0f}, trend={fold_recent_trend:.1f}/week")
    print(f"DEBUG: Recent stats for fold {fold_count}: mean={fold_recent_mean}, max={fold_recent_max}, trend={fold_recent_trend}")

    # Instantiate model with enhanced configuration
    print(f"DEBUG: Instantiating TransformerForecastModel for fold {fold_count}...")
    # model = TransformerForecastModel(...) # Original line
    current_model = None # Initialize
    try:
        current_model = TransformerForecastModel( # Renamed model
            countries=[country],
            context_length=context_length,
            prediction_length=prediction_length,
            lags_sequence=lags,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            date_col="Date_reported",
            country_col="Country",
            target_col="New_cases_scaled",
            distribution_output='negative_binomial',
            embedding_dim=64,
            num_parallel_samples=200,
            log_transform=False,
            clip_negative=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            num_time_features=num_time_features,
            d_model=128,
            encoder_layers=3,
            decoder_layers=3,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            dropout=0.15
        )
        print(f"DEBUG: TransformerForecastModel instantiated. Device: {current_model.device if hasattr(current_model, 'device') else 'N/A'}")
    except NameError as ne_tfm:
        print(f"DEBUG: FATAL ERROR - TransformerForecastModel is not defined. Ensure it's imported/defined. Error: {ne_tfm}")
        raise
    except Exception as e_model_instantiation:
        print(f"DEBUG: FATAL ERROR - Could not instantiate TransformerForecastModel: {e_model_instantiation}")
        raise

    # Set custom time features function
    print(f"DEBUG: Setting time features function on current_model for fold {fold_count}...")
    # model.set_time_features_fn(create_time_features) # Original line
    current_model.set_time_features_fn(create_time_features)
    print("DEBUG: Custom time features function set on current_model.")

    # Initialize for this fold's scope
    fold_forecast_values = np.array([])
    fold_lower_90_ci = None
    fold_upper_90_ci = None
    fold_pred_df_output = pd.DataFrame()

    try:
        # Train model
        print(f"DEBUG: Starting current_model.train() for fold {fold_count}...")
        # print(f"   Training model...") # Original print
        print(f"   Training model...")
        # train_info = model.train(train_df) # Original line
        current_train_info = current_model.train(current_train_df) # Renamed
        print(f"DEBUG: current_model.train() completed. current_train_info: {current_train_info}")
        # print(f"   Training completed. Loss: {train_info.get('training_loss', 'N/A')}") # Original print
        print(f"   Training completed. Loss: {current_train_info.get('training_loss', 'N/A')}")

        # Generate predictions
        # start_date = test_df['Date_reported'].iloc[0] # Original line
        fold_prediction_start_date = current_test_df['Date_reported'].iloc[0] # Renamed
        print(f"DEBUG: Calling current_model.predict() for fold {fold_count}. Start date: {fold_prediction_start_date}, prediction_length: {prediction_length}")
        # pred_df = model.predict(country, start_date, prediction_length) # Original line
        fold_pred_df_output = current_model.predict(country, fold_prediction_start_date, prediction_length) # Renamed
        print(f"DEBUG: current_model.predict() completed. fold_pred_df_output shape: {fold_pred_df_output.shape}")
        if not fold_pred_df_output.empty:
            print(f"DEBUG: fold_pred_df_output.head():\n{fold_pred_df_output.head()}")

        # Inverse transform predictions
        # scaled_forecast = pred_df['forecast'].values # Original line
        fold_scaled_forecast = fold_pred_df_output['forecast'].values # Renamed
        print(f"DEBUG: fold_scaled_forecast (from model, first 5): {fold_scaled_forecast[:5] if len(fold_scaled_forecast)>0 else 'N/A'}")
        # forecast = inverse_transform(scaled_forecast) # Original line
        fold_forecast_values = inverse_transform_fold_fn(fold_scaled_forecast) # Renamed
        print(f"DEBUG: fold_forecast_values (inverse transformed, first 5): {fold_forecast_values[:5] if len(fold_forecast_values)>0 else 'N/A'}")

        # Post-processing adjustments
        # 1. Ensure non-negative
        # forecast = np.maximum(0, forecast) # Original line
        print(f"DEBUG: fold_forecast_values before non-negative adj (sum of neg): {np.sum(fold_forecast_values[fold_forecast_values < 0]) if len(fold_forecast_values)>0 else 0}")
        fold_forecast_values = np.maximum(0, fold_forecast_values)
        print(f"DEBUG: fold_forecast_values after non-negative adj (first 5): {fold_forecast_values[:5] if len(fold_forecast_values)>0 else 'N/A'}")

        # 2. Apply reality checks based on recent data
        # max_allowed = recent_max * 5 # Original line
        fold_max_allowed = fold_recent_max * 5 # Renamed
        print(f"DEBUG: fold_max_allowed for capping: {fold_max_allowed} (fold_recent_max was {fold_recent_max})")
        # forecast = np.minimum(forecast, max_allowed) # Original line
        print(f"DEBUG: fold_forecast_values before capping (max value): {np.max(fold_forecast_values) if len(fold_forecast_values)>0 else 'N/A'}")
        fold_forecast_values = np.minimum(fold_forecast_values, fold_max_allowed)
        print(f"DEBUG: fold_forecast_values after capping (first 5): {fold_forecast_values[:5] if len(fold_forecast_values)>0 else 'N/A'}, (max value): {np.max(fold_forecast_values) if len(fold_forecast_values)>0 else 'N/A'}")

        # 3. Round to integers
        # forecast = np.round(forecast).astype(int) # Original line
        print(f"DEBUG: fold_forecast_values before rounding (first 5, sum of decimals): {np.sum(fold_forecast_values - np.floor(fold_forecast_values)) if len(fold_forecast_values)>0 else 0}")
        fold_forecast_values = np.round(fold_forecast_values).astype(int)
        print(f"DEBUG: fold_forecast_values after rounding (first 5): {fold_forecast_values[:5] if len(fold_forecast_values)>0 else 'N/A'}")

        # Get confidence intervals if available
        # if 'lower_90' in pred_df.columns: # Original line
        if 'lower_90' in fold_pred_df_output.columns and 'upper_90' in fold_pred_df_output.columns: # Check both
            print("DEBUG: Processing confidence intervals from fold_pred_df_output.")
            # lower_90 = np.maximum(0, inverse_transform(pred_df['lower_90'].values)) # Original line
            # upper_90 = inverse_transform(pred_df['upper_90'].values) # Original line
            # upper_90 = np.minimum(upper_90, max_allowed * 2) # Original line

            scaled_lower_90 = fold_pred_df_output['lower_90'].values
            scaled_upper_90 = fold_pred_df_output['upper_90'].values
            print(f"DEBUG: scaled_lower_90 (first 5): {scaled_lower_90[:5] if len(scaled_lower_90)>0 else 'N/A'}")
            print(f"DEBUG: scaled_upper_90 (first 5): {scaled_upper_90[:5] if len(scaled_upper_90)>0 else 'N/A'}")

            fold_lower_90_ci = np.maximum(0, inverse_transform_fold_fn(scaled_lower_90)) # Renamed
            fold_upper_90_ci = inverse_transform_fold_fn(scaled_upper_90) # Renamed

            print(f"DEBUG: fold_lower_90_ci (inv. transformed, first 5): {fold_lower_90_ci[:5] if len(fold_lower_90_ci)>0 else 'N/A'}")
            print(f"DEBUG: fold_upper_90_ci (inv. transformed, first 5): {fold_upper_90_ci[:5] if len(fold_upper_90_ci)>0 else 'N/A'}")

            fold_upper_90_ci = np.minimum(fold_upper_90_ci, fold_max_allowed * 2)
            print(f"DEBUG: fold_upper_90_ci after capping (first 5): {fold_upper_90_ci[:5] if len(fold_upper_90_ci)>0 else 'N/A'}")
        else:
            print("DEBUG: CI columns ('lower_90' or 'upper_90') not found in fold_pred_df_output.")
            fold_lower_90_ci = None
            fold_upper_90_ci = None

    except Exception as e_model_ops: # Renamed 'e'
        # print(f"   Model training/prediction failed: {e}") # Original print
        print(f"DEBUG: ERROR during model training/prediction ops for fold {fold_count}: {e_model_ops}")
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}")
        print(f"   Model training/prediction failed: {e_model_ops}")
        continue

    # Get actual values
    # actual = test_df['New_cases'].iloc[:prediction_length].values # Original line
    fold_actual_values = current_test_df['New_cases'].iloc[:prediction_length].values # Renamed
    print(f"DEBUG: fold_actual_values (for metrics, length {len(fold_actual_values)}, first 5): {fold_actual_values[:5] if len(fold_actual_values)>0 else 'N/A'}")

    # Ensure forecast has same length as actuals for metrics
    if len(fold_forecast_values) != len(fold_actual_values):
        print(f"DEBUG: WARNING - Length mismatch for fold {fold_count}: fold_forecast_values ({len(fold_forecast_values)}) vs fold_actual_values ({len(fold_actual_values)}). Adjusting forecast for metrics.")
        fold_forecast_values = fold_forecast_values[:len(fold_actual_values)]
        if fold_lower_90_ci is not None: fold_lower_90_ci = fold_lower_90_ci[:len(fold_actual_values)]
        if fold_upper_90_ci is not None: fold_upper_90_ci = fold_upper_90_ci[:len(fold_actual_values)]
        print(f"DEBUG: Adjusted fold_forecast_values to length {len(fold_forecast_values)}.")

    # Calculate comprehensive metrics
    print(f"DEBUG: Calculating metrics for fold {fold_count}...")
    # mae = np.mean(np.abs(actual - forecast)) # Original line
    metric_mae = np.mean(np.abs(fold_actual_values - fold_forecast_values)) if len(fold_actual_values) > 0 else np.nan
    # mse = np.mean((actual - forecast) ** 2) # Original line
    metric_mse = np.mean((fold_actual_values - fold_forecast_values) ** 2) if len(fold_actual_values) > 0 else np.nan
    # rmse = np.sqrt(mse) # Original line
    metric_rmse = np.sqrt(metric_mse) if not pd.isna(metric_mse) else np.nan

    epsilon_metric_val = 1.0 # Renamed epsilon for metrics
    metric_mape, metric_smape, metric_log_acc = np.nan, np.nan, np.nan # Initialize

    if len(fold_actual_values) > 0:
        # mape = np.mean(np.abs((actual - forecast) / (actual + epsilon))) * 100 # Original line
        safe_denom_mape = fold_actual_values + epsilon_metric_val
        safe_denom_mape[safe_denom_mape == 0] = epsilon_metric_val
        metric_mape = np.mean(np.abs((fold_actual_values - fold_forecast_values) / safe_denom_mape)) * 100

        # smape = np.mean(2 * np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast) + epsilon)) * 100 # Original line
        safe_denom_smape = np.abs(fold_actual_values) + np.abs(fold_forecast_values) + epsilon_metric_val
        safe_denom_smape[safe_denom_smape == 0] = epsilon_metric_val
        metric_smape = np.mean(2 * np.abs(fold_actual_values - fold_forecast_values) / safe_denom_smape) * 100

        # log_acc = np.mean(np.abs(np.log1p(actual) - np.log1p(forecast))) # Original line
        metric_log_acc = np.mean(np.abs(np.log1p(fold_actual_values) - np.log1p(fold_forecast_values)))
    else:
        print("DEBUG: fold_actual_values is empty for metrics. Metrics (mape, smape, log_acc) set to NaN.")

    # Coverage if we have intervals
    metric_coverage = np.nan # Renamed
    # if lower_90 is not None and upper_90 is not None: # Original line
    #     coverage = np.mean((actual >= lower_90) & (actual <= upper_90)) * 100 # Original line
    if fold_lower_90_ci is not None and fold_upper_90_ci is not None and len(fold_actual_values) > 0:
        if len(fold_lower_90_ci) == len(fold_actual_values) and len(fold_upper_90_ci) == len(fold_actual_values):
            metric_coverage = np.mean((fold_actual_values >= fold_lower_90_ci) & (fold_actual_values <= fold_upper_90_ci)) * 100
            print(f"DEBUG: Calculated 90% Coverage for fold {fold_count}: {metric_coverage:.1f}%")
        else:
            print(f"DEBUG: Coverage not calculated for fold {fold_count} due to CI/actual length mismatch.")
    # else: # Original line (coverage = np.nan)
    #     coverage = np.nan # Original line

    print(f"DEBUG: Metrics for fold {fold_count}: MAE={metric_mae:.0f}, RMSE={metric_rmse:.0f}, MAPE={metric_mape:.1f}%, SMAPE={metric_smape:.1f}%, LogAcc={metric_log_acc:.3f}, Coverage={metric_coverage:.1f}%")

    # Original prints:
    # print(f"\n   Metrics:")
    # print(f"     MAE:  {mae:,.0f}") ...
    print(f"\n   Metrics:")
    print(f"     MAE:  {metric_mae:,.0f}")
    print(f"     RMSE: {metric_rmse:,.0f}")
    print(f"     MAPE: {metric_mape:.1f}%")
    print(f"     SMAPE: {metric_smape:.1f}%")
    print(f"     Log Accuracy: {metric_log_acc:.3f}")
    if not np.isnan(metric_coverage):
        print(f"     90% Coverage: {metric_coverage:.1f}%")

    # Enhanced visualization
    print(f"DEBUG: Preparing plot for fold {fold_count}...")
    plt.figure(figsize=(14, 6))

    # hist_weeks = 26 # Original line
    # hist_start_idx = max(0, train_idx[-1] - hist_weeks) # Original line
    # hist_df = series_df.iloc[hist_start_idx:train_idx[-1] + 1] # Original line
    plot_hist_weeks = 26
    plot_hist_end_idx = train_indices_cv[-1]
    plot_hist_start_idx = max(0, plot_hist_end_idx - plot_hist_weeks + 1)
    plot_hist_df = series_df.iloc[plot_hist_start_idx : plot_hist_end_idx + 1]
    print(f"DEBUG: Plotting history from index {plot_hist_start_idx} to {plot_hist_end_idx}. plot_hist_df shape: {plot_hist_df.shape}")

    if not plot_hist_df.empty:
        # plt.plot(hist_df['Date_reported'], hist_df['New_cases'], ...) # Original line
        plt.plot(plot_hist_df['Date_reported'], plot_hist_df['New_cases'],
                 'b-', label='Historical', alpha=0.7, linewidth=1.5)

    # test_dates = test_df['Date_reported'].iloc[:prediction_length] # Original line
    # plt.plot(test_dates, actual, ...) # Original line
    plot_actual_val_dates = current_test_df['Date_reported'].iloc[:prediction_length]
    if len(plot_actual_val_dates) == len(fold_actual_values):
        plt.plot(plot_actual_val_dates, fold_actual_values,
                 'go-', label='Actual', markersize=8, linewidth=2)
    else:
        print(f"DEBUG: Plot actuals: Mismatch dates ({len(plot_actual_val_dates)}) vs values ({len(fold_actual_values)})")


    # plt.plot(pred_df['date'], forecast, ...) # Original line
    if not fold_pred_df_output.empty and 'date' in fold_pred_df_output.columns:
        plot_forecast_val_dates = fold_pred_df_output['date'].iloc[:len(fold_forecast_values)]
        if len(plot_forecast_val_dates) == len(fold_forecast_values):
            plt.plot(plot_forecast_val_dates, fold_forecast_values,
                     'rs--', label='Forecast', markersize=8, linewidth=2)
        else:
            print(f"DEBUG: Plot forecast: Mismatch dates ({len(plot_forecast_val_dates)}) vs values ({len(fold_forecast_values)})")


    # if lower_90 is not None and upper_90 is not None: # Original line
    #     plt.fill_between(pred_df['date'], lower_90, upper_90, ...) # Original line
    if fold_lower_90_ci is not None and fold_upper_90_ci is not None and not fold_pred_df_output.empty:
        plot_ci_val_dates = fold_pred_df_output['date'].iloc[:len(fold_lower_90_ci)]
        if len(plot_ci_val_dates) == len(fold_lower_90_ci) and len(plot_ci_val_dates) == len(fold_upper_90_ci):
             plt.fill_between(plot_ci_val_dates, fold_lower_90_ci, fold_upper_90_ci,
                             color='red', alpha=0.2, label='90% Prediction Interval')
        else:
            print(f"DEBUG: Plot CIs: Mismatch dates vs CI values lengths.")


    # plt.axvline(start_date, color='black', ...) # Original line (start_date was fold_prediction_start_date)
    plt.axvline(fold_prediction_start_date, color='black', linestyle='--', alpha=0.5, label='Forecast Start')

    # textstr = f'MAE: {mae:,.0f}\nRMSE: {rmse:,.0f}\nSMAPE: {smape:.1f}%' # Original line
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8) # Original line
    # plt.text(0.02, 0.98, textstr, ...) # Original line
    plot_textstr_metrics = f'MAE: {metric_mae:,.0f}\nRMSE: {metric_rmse:,.0f}\nSMAPE: {metric_smape:.1f}%'
    plot_props_metrics = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, plot_textstr_metrics, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=plot_props_metrics)

    # plt.title(f'Fold {fold} - {country} COVID-19 Forecast', fontsize=14) # Original line
    plt.title(f'Fold {fold_count} - {country} COVID-19 Forecast', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('New Cases')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # if hist_df['New_cases'].max() / (hist_df['New_cases'][hist_df['New_cases'] > 0].min() + 1) > 100: # Original line
    #     plt.yscale('symlog') # Original line
    if not plot_hist_df.empty and 'New_cases' in plot_hist_df and plot_hist_df['New_cases'].max() > 0:
        min_pos_hist = plot_hist_df['New_cases'][plot_hist_df['New_cases'] > 0].min()
        if not pd.isna(min_pos_hist) and min_pos_hist > 0 and \
           (plot_hist_df['New_cases'].max() / min_pos_hist) > 100:
            plt.yscale('symlog')
            print(f"DEBUG: Using symlog scale for y-axis on plot for fold {fold_count}.")

    plt.tight_layout()
    plt.show()

    # Store results
    print(f"DEBUG: Storing metrics for fold {fold_count} to metrics_results.")
    # metrics.append({...}) # Original line
    metrics_results.append({
        'fold': fold_count,
        'train_start': current_train_df['Date_reported'].iloc[0],
        'train_end': current_train_df['Date_reported'].iloc[-1],
        'test_start': plot_actual_val_dates.iloc[0] if not plot_actual_val_dates.empty else pd.NaT,
        'test_end': plot_actual_val_dates.iloc[-1] if not plot_actual_val_dates.empty else pd.NaT,
        'mae': metric_mae,
        'mse': metric_mse,
        'rmse': metric_rmse,
        'mape': metric_mape,
        'smape': metric_smape,
        'log_accuracy': metric_log_acc,
        'coverage_90': metric_coverage,
        'scaler_type': type(active_scaler).__name__ if active_scaler != 'log' else 'log',
        'mean_actual': np.mean(fold_actual_values) if len(fold_actual_values) > 0 else np.nan,
        'mean_forecast': np.mean(fold_forecast_values) if len(fold_forecast_values) > 0 else np.nan,
        'recent_train_mean': fold_recent_mean,
        'experiment_description': experiment_desc
    })

    # Store predictions for ensemble analysis
    print(f"DEBUG: Storing predictions for fold {fold_count} to fold_predictions_data.")
    # fold_predictions.append({...}) # Original line
    fold_predictions_data.append({
        'fold': fold_count,
        'dates': plot_forecast_val_dates.values if 'plot_forecast_val_dates' in locals() and not plot_forecast_val_dates.empty else np.array([]),
        'actual': fold_actual_values,
        'forecast': fold_forecast_values,
        'lower_90': fold_lower_90_ci,
        'upper_90': fold_upper_90_ci
    })
    print(f"DEBUG: ----- End of FOLD {fold_count} processing -----")

print("DEBUG: CV folds loop finished.")
# Compile results
# current_df = pd.DataFrame(metrics) # Original line
final_metrics_df = pd.DataFrame(metrics_results) # Renamed
print(f"DEBUG: Compiled results into final_metrics_df. Shape: {final_metrics_df.shape}")
if not final_metrics_df.empty:
    print(f"DEBUG: final_metrics_df.head():\n{final_metrics_df.head()}")

# if len(current_df) > 0: # Original line
if not final_metrics_df.empty:
    # Original prints:
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    print(f"\nTotal folds completed: {len(final_metrics_df)}")
    print(f"DEBUG: Calculating overall performance statistics from final_metrics_df.")
    print(f"\nOverall Performance:")
    print(f"   MAE:  {final_metrics_df['mae'].mean():,.0f} ± {final_metrics_df['mae'].std():,.0f}")
    print(f"   RMSE: {final_metrics_df['rmse'].mean():,.0f} ± {final_metrics_df['rmse'].std():,.0f}")
    print(f"   SMAPE: {final_metrics_df['smape'].mean():.1f}% ± {final_metrics_df['smape'].std():.1f}%")
    print(f"   Log Accuracy: {final_metrics_df['log_accuracy'].mean():.3f} ± {final_metrics_df['log_accuracy'].std():.3f}")

    if 'coverage_90' in final_metrics_df.columns:
        # coverage_mean = current_df['coverage_90'].dropna().mean() # Original line
        summary_coverage_mean = final_metrics_df['coverage_90'].dropna().mean() # Renamed
        if not np.isnan(summary_coverage_mean):
            # print(f"   90% Coverage: {coverage_mean:.1f}%") # Original line
            print(f"   90% Coverage: {summary_coverage_mean:.1f}%")
            print(f"DEBUG: Mean 90% Coverage (summary): {summary_coverage_mean:.1f}%")

    # Performance by scaler type
    print(f"\nPerformance by Scaler Type:")
    if 'scaler_type' in final_metrics_df.columns:
        # for scaler_type in current_df['scaler_type'].unique(): # Original line
        for summary_scaler_type in final_metrics_df['scaler_type'].unique(): # Renamed
            # scaler_df = current_df[current_df['scaler_type'] == scaler_type] # Original line
            summary_scaler_df = final_metrics_df[final_metrics_df['scaler_type'] == summary_scaler_type] # Renamed
            print(f"DEBUG: Performance for scaler type (summary): {summary_scaler_type}, {len(summary_scaler_df)} folds.")
            # print(f"\n   {scaler_type}:") # Original line
            print(f"\n   {summary_scaler_type}:")
            # print(f"     Folds: {len(scaler_df)}") # Original line
            print(f"     Folds: {len(summary_scaler_df)}")
            # print(f"     SMAPE: {scaler_df['smape'].mean():.1f}% ± {scaler_df['smape'].std():.1f}%") # Original line
            print(f"     SMAPE: {summary_scaler_df['smape'].mean():.1f}% ± {summary_scaler_df['smape'].std():.1f}%")
    else:
        print("DEBUG: 'scaler_type' column not found in final_metrics_df.")


    # Plot summary of errors over time
    print("DEBUG: Preparing summary plots based on final_metrics_df.")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    # plt.plot(range(1, len(current_df) + 1), current_df['smape'], 'o-') # Original line
    if 'smape' in final_metrics_df.columns and not final_metrics_df['smape'].empty:
        plt.plot(range(1, len(final_metrics_df) + 1), final_metrics_df['smape'], 'o-')
    else:
        print("DEBUG: SMAPE data not available for summary plot 1.")
    plt.xlabel('Fold')
    plt.ylabel('SMAPE (%)')
    plt.title('SMAPE by Fold')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # plt.scatter(current_df['mean_actual'], current_df['mean_forecast'], alpha=0.7) # Original line
    # max_val = max(current_df['mean_actual'].max(), current_df['mean_forecast'].max()) # Original line
    # plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5) # Original line

    summary_plot_max_val = 0
    if 'mean_actual' in final_metrics_df.columns and 'mean_forecast' in final_metrics_df.columns and \
       not final_metrics_df['mean_actual'].empty and not final_metrics_df['mean_forecast'].empty:

        plt.scatter(final_metrics_df['mean_actual'], final_metrics_df['mean_forecast'], alpha=0.7)

        summary_actual_max = final_metrics_df['mean_actual'].max(skipna=True)
        summary_forecast_max = final_metrics_df['mean_forecast'].max(skipna=True)

        if not pd.isna(summary_actual_max) and not pd.isna(summary_forecast_max):
            summary_plot_max_val = max(summary_actual_max, summary_forecast_max)
        elif not pd.isna(summary_actual_max):
            summary_plot_max_val = summary_actual_max
        elif not pd.isna(summary_forecast_max):
            summary_plot_max_val = summary_forecast_max
        if pd.isna(summary_plot_max_val): summary_plot_max_val = 0
    else:
        print("DEBUG: Mean actual or mean forecast data not available for summary scatter plot.")

    plt.plot([0, summary_plot_max_val], [0, summary_plot_max_val], 'r--', alpha=0.5)
    plt.xlabel('Mean Actual')
    plt.ylabel('Mean Forecast')
    plt.title('Mean Actual vs Mean Forecast')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Save results
# output_csv = 'enhanced_expanding_window_results.csv' # Original line
output_filename_csv = 'enhanced_expanding_window_results.csv' # Renamed
print(f"DEBUG: Output CSV file name for saving: {output_filename_csv}")

try:
    print("DEBUG: Attempting to load and combine with existing results for saving...")
    # df_combined = current_df # Default if no history, original line
    df_to_save_combined = final_metrics_df # Default if no history

    # if os.path.exists(output_csv): # Original line
    if os.path.exists(output_filename_csv):
        print(f"DEBUG: Existing results file found: {output_filename_csv}. Loading...")
        try:
            # history_df = pd.read_csv(output_csv, parse_dates=['train_start', 'train_end', 'test_start', 'test_end']) # Original line
            csv_history_df = pd.read_csv(output_filename_csv, parse_dates=['train_start', 'train_end', 'test_start', 'test_end']) # Renamed
        except ValueError as ve_load_csv:
            print(f"DEBUG: Warning - Could not parse dates from {output_filename_csv}, loading without parse_dates. Error: {ve_load_csv}")
            csv_history_df = pd.read_csv(output_filename_csv)
        except Exception as e_load_csv_final:
            print(f"DEBUG: Error loading existing CSV {output_filename_csv}: {e_load_csv_final}. Will use current results only.")
            csv_history_df = pd.DataFrame() # Empty df if load fails

        print(f"DEBUG: Loaded csv_history_df. Shape: {csv_history_df.shape}")
        if not final_metrics_df.empty: # Only concat if current results exist
            # df_combined = pd.concat([history_df, current_df], ignore_index=True) # Original line
            df_to_save_combined = pd.concat([csv_history_df, final_metrics_df], ignore_index=True)
            print(f"DEBUG: Combined csv_history_df and final_metrics_df. New shape: {df_to_save_combined.shape}")
        elif not csv_history_df.empty: # If current is empty but history exists
             df_to_save_combined = csv_history_df
             print("DEBUG: final_metrics_df is empty, df_to_save_combined is the same as csv_history_df.")
        # If both are empty, df_to_save_combined remains empty (initialised from final_metrics_df which would be empty)
    # else: # Original line (df_combined = current_df)
        # df_combined = current_df # This is already handled by initialization of df_to_save_combined

    if not df_to_save_combined.empty:
        # df_combined.to_csv(output_csv, index=False) # Original line
        df_to_save_combined.to_csv(output_filename_csv, index=False)
        # print(f"\nResults saved to {output_csv}") # Original print
        print(f"\nResults saved to {output_filename_csv}")
        print(f"DEBUG: Combined results (shape {df_to_save_combined.shape}) saved to {output_filename_csv}.")
    else:
        print(f"DEBUG: df_to_save_combined is empty. Nothing to save to {output_filename_csv}.")

    # if fold_predictions: # Original line
    if fold_predictions_data:
        import pickle
        # with open('fold_predictions.pkl', 'wb') as f: # Original line
        #     pickle.dump(fold_predictions, f) # Original line
        output_filename_pkl = 'fold_predictions.pkl' # Renamed
        print(f"DEBUG: Saving detailed predictions (from fold_predictions_data) to {output_filename_pkl}...")
        with open(output_filename_pkl, 'wb') as f_out_pkl: # Renamed
            pickle.dump(fold_predictions_data, f_out_pkl)
        # print(f"Detailed predictions saved to fold_predictions.pkl") # Original print
        print(f"Detailed predictions saved to {output_filename_pkl}")
        print(f"DEBUG: Detailed predictions saved. Items: {len(fold_predictions_data)}")
    else:
        print("DEBUG: fold_predictions_data is empty. No detailed predictions to save.")

except Exception as e_final_save: # Renamed 'e'
    # print(f"Error saving results: {e}") # Original print
    print(f"DEBUG: ERROR saving results: {e_final_save}")
    import traceback
    print(f"DEBUG: Traceback for saving error: {traceback.format_exc()}")
    print(f"Error saving results: {e_final_save}")

# print("\nExperiment completed!") # Original print
print("\nExperiment completed!")
print("DEBUG: ====== Script execution finished. ======")

plt.tight_layout()
plt.show()
input('Press Enter to exit and close all plots...')
