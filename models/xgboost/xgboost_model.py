import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


class XGBoostForecastModel:
    """
    XGBoost model for time series forecasting that matches the TransformerForecastModel interface.
    """
    
    def __init__(
        self,
        countries: List[str],
        date_col: str = "Date_reported",
        country_col: str = "Country", 
        target_col: str = "New_cases",
        context_length: int = 48,
        prediction_length: int = 4,
        lags_sequence: List[int] = None,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        n_estimators: int = 200,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 1.0,
        reg_lambda: float = 1.0,
        log_transform: bool = True,
        include_datetime_features: bool = True,
        include_rolling_features: bool = True,
        rolling_windows: List[int] = None,
        device: str = 'cpu',
        **kwargs
    ):
        self.countries = countries
        self.date_col = date_col
        self.country_col = country_col
        self.target_col = target_col
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.log_transform = log_transform
        self.include_datetime_features = include_datetime_features
        self.include_rolling_features = include_rolling_features
        
        # XGBoost parameters
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        
        # Default lags sequence if none provided
        if lags_sequence is None:
            lags_sequence = [1, 2, 4, 8, 12, 24]  # More granular lags for better patterns
        
        # Filter out lags that are too long for context
        self.lags_sequence = [lag for lag in lags_sequence if lag <= context_length]
        if len(self.lags_sequence) < len(lags_sequence):
            dropped = set(lags_sequence) - set(self.lags_sequence)
            print(f"Warning: dropping lags > context_length: {dropped}")
            
        self.max_lag = max(self.lags_sequence) if self.lags_sequence else 1
        
        # Rolling windows for features
        if rolling_windows is None:
            rolling_windows = [2, 4, 8, 12, 24]  # Better suited for weekly data patterns
        self.rolling_windows = [w for w in rolling_windows if w <= context_length]
        
        # Initialize models (one for each prediction step for multi-step forecasting)
        self.models = {}
        self._train_df = None
        self._feature_names = None
        
        print(f"XGBoost model initialized for {len(countries)} countries")
        print(f"Lags: {self.lags_sequence}")
        print(f"Rolling windows: {self.rolling_windows}")

    def _create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create datetime-based features"""
        if not self.include_datetime_features:
            return df
            
        df = df.copy()
        dates = pd.to_datetime(df[self.date_col])
        
        # Basic datetime features
        df['hour'] = dates.dt.hour
        df['day'] = dates.dt.day
        df['month'] = dates.dt.month
        df['quarter'] = dates.dt.quarter
        df['dayofweek'] = dates.dt.dayofweek
        df['dayofyear'] = dates.dt.dayofyear
        df['week'] = dates.dt.isocalendar().week
        
        # Binary features
        df['is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
        df['is_month_start'] = dates.dt.is_month_start.astype(int)
        df['is_month_end'] = dates.dt.is_month_end.astype(int)
        
        # Cyclical encoding for periodic features
        df['month_sin'] = np.sin(2 * np.pi * dates.dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * dates.dt.month / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)
        
        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features from target variable"""
        df = df.copy()
        
        # Create lag features grouped by country to avoid leakage across countries
        if self.country_col in df.columns:
            for lag in self.lags_sequence:
                df[f'lag_{lag}'] = df.groupby(self.country_col)[self.target_col].shift(lag)
        else:
            # If no country column, create lags normally
            for lag in self.lags_sequence:
                df[f'lag_{lag}'] = df[self.target_col].shift(lag)
            
        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistics"""
        if not self.include_rolling_features:
            return df
            
        df = df.copy()
        
        # Create rolling features grouped by country to avoid leakage across countries
        if self.country_col in df.columns:
            for window in self.rolling_windows:
                try:
                    # Rolling statistics per country using transform to maintain index alignment
                    df[f'rolling_mean_{window}'] = df.groupby(self.country_col)[self.target_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean())
                    df[f'rolling_std_{window}'] = df.groupby(self.country_col)[self.target_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std())
                    df[f'rolling_min_{window}'] = df.groupby(self.country_col)[self.target_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).min())
                    df[f'rolling_max_{window}'] = df.groupby(self.country_col)[self.target_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).max())
                    
                    # Fill any NaN in rolling std with 0
                    df[f'rolling_std_{window}'] = df[f'rolling_std_{window}'].fillna(0)
                    
                    # Trend features (difference from rolling mean)
                    df[f'trend_{window}'] = df[self.target_col] - df[f'rolling_mean_{window}']
                    
                    # Additional momentum and volatility features
                    df[f'momentum_{window}'] = df.groupby(self.country_col)[self.target_col].transform(
                        lambda x: x.pct_change(periods=window)).fillna(0)
                    df[f'volatility_{window}'] = df[f'rolling_std_{window}'] / (df[f'rolling_mean_{window}'] + 1e-8)
                    df[f'range_{window}'] = df[f'rolling_max_{window}'] - df[f'rolling_min_{window}']
                    
                    # Weekly relative position
                    df[f'position_{window}'] = (df[self.target_col] - df[f'rolling_min_{window}']) / (df[f'range_{window}'] + 1e-8)
                    
                except Exception as e:
                    print(f"Warning: Failed to create rolling features for window {window}: {e}")
                    # Fallback to simple rolling without grouping
                    df[f'rolling_mean_{window}'] = df[self.target_col].rolling(window=window, min_periods=1).mean()
                    df[f'rolling_std_{window}'] = df[self.target_col].rolling(window=window, min_periods=1).std().fillna(0)
                    df[f'rolling_min_{window}'] = df[self.target_col].rolling(window=window, min_periods=1).min()
                    df[f'rolling_max_{window}'] = df[self.target_col].rolling(window=window, min_periods=1).max()
                    df[f'trend_{window}'] = df[self.target_col] - df[f'rolling_mean_{window}']
                    df[f'momentum_{window}'] = df[self.target_col].pct_change(periods=window).fillna(0)
                    df[f'volatility_{window}'] = df[f'rolling_std_{window}'] / (df[f'rolling_mean_{window}'] + 1e-8)
                    df[f'range_{window}'] = df[f'rolling_max_{window}'] - df[f'rolling_min_{window}']
                    df[f'position_{window}'] = (df[self.target_col] - df[f'rolling_min_{window}']) / (df[f'range_{window}'] + 1e-8)
        else:
            # If no country column, create rolling features normally
            for window in self.rolling_windows:
                # Rolling statistics
                df[f'rolling_mean_{window}'] = df[self.target_col].rolling(window=window, min_periods=1).mean()
                df[f'rolling_std_{window}'] = df[self.target_col].rolling(window=window, min_periods=1).std().fillna(0)
                df[f'rolling_min_{window}'] = df[self.target_col].rolling(window=window, min_periods=1).min()
                df[f'rolling_max_{window}'] = df[self.target_col].rolling(window=window, min_periods=1).max()
                
                # Trend features (difference from rolling mean)
                df[f'trend_{window}'] = df[self.target_col] - df[f'rolling_mean_{window}']
                
                # Additional features
                df[f'momentum_{window}'] = df[self.target_col].pct_change(periods=window).fillna(0)
                df[f'volatility_{window}'] = df[f'rolling_std_{window}'] / (df[f'rolling_mean_{window}'] + 1e-8)
                df[f'range_{window}'] = df[f'rolling_max_{window}'] - df[f'rolling_min_{window}']
                df[f'position_{window}'] = (df[self.target_col] - df[f'rolling_min_{window}']) / (df[f'range_{window}'] + 1e-8)
            
        return df

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features for the model"""
        # Sort by country and date to ensure proper lag calculation
        if self.country_col in df.columns:
            df = df.sort_values([self.country_col, self.date_col]).reset_index(drop=True)
        else:
            df = df.sort_values(self.date_col).reset_index(drop=True)
        
        # Apply log transform if specified
        if self.log_transform:
            df[self.target_col] = np.log1p(np.maximum(0, df[self.target_col]))
        
        # Create features
        df = self._create_datetime_features(df)
        df = self._create_lag_features(df)
        df = self._create_rolling_features(df)
        
        # Add some additional derived features
        if len(df) > 1:
            # First difference (change from previous period)
            if self.country_col in df.columns:
                df['first_diff'] = df.groupby(self.country_col)[self.target_col].diff().fillna(0)
            else:
                df['first_diff'] = df[self.target_col].diff().fillna(0)
            
            # Boolean indicators for direction
            df['is_increasing'] = (df['first_diff'] > 0).astype(int)
            df['is_decreasing'] = (df['first_diff'] < 0).astype(int)
        
        return df

    def _prepare_training_data(self, df: pd.DataFrame):
        """Prepare data for training with multi-step ahead targets"""
        # Create features for all countries together
        df_features = self._create_features(df)
        
        # Drop rows with NaN values (due to lags and rolling windows)
        # This should be done per country to avoid dropping valid data
        min_required_rows = max(self.max_lag, max(self.rolling_windows) if self.rolling_windows else 0)
        
        if self.country_col in df_features.columns:
            # Filter out initial rows per country that don't have enough lag data
            valid_rows = []
            for country in df_features[self.country_col].unique():
                country_data = df_features[df_features[self.country_col] == country]
                if len(country_data) > min_required_rows:
                    valid_rows.append(country_data.iloc[min_required_rows:])
            
            if valid_rows:
                df_features = pd.concat(valid_rows, ignore_index=True)
            else:
                raise ValueError(f"No country has enough data points (need > {min_required_rows})")
        else:
            df_features = df_features.iloc[min_required_rows:].copy()
        
        # Identify feature columns (exclude metadata columns)
        exclude_cols = {self.date_col, self.country_col, self.target_col}
        feature_cols = [col for col in df_features.columns if col not in exclude_cols]
        self._feature_names = feature_cols
        
        # Ensure all features are numeric
        X = df_features[feature_cols].copy()
        
        # Convert any remaining object/string columns to numeric
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'string':
                # Try to convert to numeric, fill NaN with 0
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Fill any remaining NaN values with 0
        X = X.fillna(0)
        
        # Convert to numpy array and ensure proper dtype
        X = X.astype(np.float64).values
        
        # Handle infinite values that can cause XGBoost errors
        X = np.where(np.isinf(X), 0, X)
        X = np.where(np.isnan(X), 0, X)
        
        # Clip extreme values to prevent numerical issues
        X = np.clip(X, -1e10, 1e10)
        
        # Create targets for multi-step forecasting
        y_dict = {}
        for step in range(1, self.prediction_length + 1):
            if self.country_col in df_features.columns:
                # Create targets grouped by country to avoid leakage
                try:
                    y_step = df_features.groupby(self.country_col)[self.target_col].transform(
                        lambda x: x.shift(-step))
                except Exception as e:
                    print(f"Warning: Failed to create grouped targets for step {step}: {e}")
                    # Fallback to regular shift
                    y_step = df_features[self.target_col].shift(-step)
            else:
                y_step = df_features[self.target_col].shift(-step)
            y_dict[step] = y_step.values
        
        # Remove rows where any target is NaN
        valid_mask = np.all([~np.isnan(y_dict[step]) for step in range(1, self.prediction_length + 1)], axis=0)
        X = X[valid_mask]
        for step in range(1, self.prediction_length + 1):
            y_dict[step] = y_dict[step][valid_mask]
            
        return X, y_dict

    def train(self, train_data: pd.DataFrame) -> dict:
        """Train the XGBoost model"""
        print("Training XGBoost model...")
        
        self._train_df = train_data.copy()
        
        # Prepare training data
        X_train, y_dict = self._prepare_training_data(train_data)
        
        if len(X_train) == 0:
            raise ValueError("No valid training samples after feature engineering. Check your data and parameters.")
        
        print(f"Training on {len(X_train)} samples with {len(self._feature_names)} features")
        
        # Train separate model for each prediction step
        training_results = {}
        
        for step in range(1, self.prediction_length + 1):
            print(f"Training model for step {step}/{self.prediction_length}")
            
            model = xgb.XGBRegressor(
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            
            model.fit(X_train, y_dict[step])
            self.models[step] = model
            
            # Calculate training metrics
            train_pred = model.predict(X_train)
            mae = mean_absolute_error(y_dict[step], train_pred)
            mse = mean_squared_error(y_dict[step], train_pred)
            
            training_results[f'step_{step}'] = {
                'mae': mae,
                'rmse': np.sqrt(mse),
                'feature_importance': dict(zip(self._feature_names, model.feature_importances_))
            }
        
        print("Training completed!")
        return training_results

    def predict(self, country: str, start_date: pd.Timestamp, horizon: int, 
                strategy: str = "recursive") -> pd.DataFrame:
        """
        Make predictions for the specified horizon
        
        Args:
            country: Country to predict for
            start_date: Start date for predictions
            horizon: Number of steps to predict ahead
            strategy: 'recursive' or 'direct' forecasting strategy
        """
        if not self.models:
            raise ValueError("Model not trained. Call train() first.")
            
        # Get historical data up to start_date for the specific country
        hist_data = self._train_df[
            (self._train_df[self.date_col] < start_date) & 
            (self._train_df[self.country_col] == country)
        ].copy()
        
        if len(hist_data) < self.max_lag:
            raise ValueError(f"Need at least {self.max_lag} historical points for prediction. Found {len(hist_data)} for {country}")
        
        # Take only the most recent context_length points for efficiency
        hist_data = hist_data.tail(self.context_length).copy()
        
        if strategy == "recursive":
            return self._predict_recursive(country, start_date, horizon, hist_data)
        elif strategy == "direct":
            return self._predict_direct(country, start_date, horizon, hist_data)
        else:
            raise ValueError("Strategy must be 'recursive' or 'direct'")
    
    def _predict_recursive(self, country: str, start_date: pd.Timestamp, 
                          horizon: int, hist_data: pd.DataFrame) -> pd.DataFrame:
        """Recursive forecasting - uses 1-step model iteratively"""
        predictions = []
        forecast_dates = []
        
        # Use only the 1-step model for recursive forecasting
        if 1 not in self.models:
            raise ValueError("No 1-step model available for recursive forecasting")
        
        model = self.models[1]
        current_data = hist_data.copy()
        
        # Determine the date frequency from historical data
        if len(current_data) >= 2:
            date_diff = (pd.to_datetime(current_data[self.date_col]).iloc[-1] - 
                        pd.to_datetime(current_data[self.date_col]).iloc[-2]).days
        else:
            date_diff = 7  # Default to weekly
        
        for step in range(horizon):
            # Generate next date based on data frequency
            next_date = start_date + timedelta(days=date_diff * step)
            forecast_dates.append(next_date)
            
            # Create a row for prediction
            pred_row = pd.DataFrame({
                self.date_col: [next_date],
                self.country_col: [country],
                self.target_col: [0]  # Placeholder, will be replaced with prediction
            })
            
            # Combine with historical data for feature creation
            extended_data = pd.concat([current_data, pred_row], ignore_index=True)
            
            # Sort by date to ensure proper feature calculation
            extended_data = extended_data.sort_values(self.date_col).reset_index(drop=True)
            
            # Create features for the extended data
            extended_features = self._create_features(extended_data)
            
            # Get features for the last row (prediction row)
            X_pred = extended_features[self._feature_names].iloc[-1:].copy()
            
            # Handle any missing values in features
            for col in X_pred.columns:
                if X_pred[col].dtype == 'object' or X_pred[col].dtype.name == 'string':
                    X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce').fillna(0)
            
            X_pred = X_pred.fillna(0).values
            
            # Make prediction using 1-step model
            pred_value = model.predict(X_pred)[0]
            
            # Inverse transform if log was applied
            if self.log_transform:
                pred_value = np.expm1(pred_value)
            
            # Ensure non-negative
            pred_value = max(0, pred_value)
            predictions.append(pred_value)
            
            # Update the prediction row with actual prediction for next iteration
            # Use the log-transformed value if log_transform is enabled
            if self.log_transform:
                pred_row[self.target_col] = np.log1p(max(0, pred_value))
            else:
                pred_row[self.target_col] = pred_value
            
            # Add the prediction to current_data for next iteration
            current_data = pd.concat([current_data, pred_row], ignore_index=True)
            
            # Keep only recent data to maintain efficiency and avoid memory issues
            if len(current_data) > self.context_length + step + 1:
                current_data = current_data.tail(self.context_length + step + 1).reset_index(drop=True)
        
        # Return results in the same format as TransformerForecastModel
        result_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': predictions
        })
        
        return result_df
    
    def _predict_direct(self, country: str, start_date: pd.Timestamp, 
                       horizon: int, hist_data: pd.DataFrame) -> pd.DataFrame:
        """Direct forecasting - uses specific models for each step simultaneously"""
        
        print(f"DEBUG: _predict_direct called with horizon={horizon}, prediction_length={self.prediction_length}")
        
        # Limit horizon to available trained models for direct forecasting
        effective_horizon = min(horizon, self.prediction_length)
        print(f"DEBUG: effective_horizon={effective_horizon}")
        
        # Determine the date frequency from historical data
        if len(hist_data) >= 2:
            date_diff = (pd.to_datetime(hist_data[self.date_col]).iloc[-1] - 
                        pd.to_datetime(hist_data[self.date_col]).iloc[-2]).days
        else:
            date_diff = 7  # Default to weekly
        
        print(f"DEBUG: date_diff={date_diff} days")
        
        # Create a dummy row representing the prediction start point
        # Use the last historical value as a better starting point than 0
        last_historical_value = hist_data[self.target_col].iloc[-1] if len(hist_data) > 0 else 0
        
        pred_row = pd.DataFrame({
            self.date_col: [start_date],
            self.country_col: [country],
            self.target_col: [last_historical_value]  # Better initialization
        })
        
        # Combine with historical data for feature creation
        extended_data = pd.concat([hist_data, pred_row], ignore_index=True)
        
        # Sort by date to ensure proper feature calculation
        extended_data = extended_data.sort_values(self.date_col).reset_index(drop=True)
        
        # Create features for the extended data
        extended_features = self._create_features(extended_data)
        
        # Get features for the last row (this represents the state at prediction time)
        X_pred = extended_features[self._feature_names].iloc[-1:].copy()
        
        # Handle any missing values in features
        for col in X_pred.columns:
            if X_pred[col].dtype == 'object' or X_pred[col].dtype.name == 'string':
                X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce').fillna(0)
        
        X_pred = X_pred.fillna(0).values
        
        # Make predictions for all steps simultaneously using different models
        predictions = []
        forecast_dates = []
        
        print(f"DEBUG: About to predict {effective_horizon} steps")
        print(f"DEBUG: Available models: {list(self.models.keys())}")
        
        for step in range(effective_horizon):
            # Generate date for this step using consistent date frequency
            next_date = start_date + timedelta(days=date_diff * step)
            forecast_dates.append(next_date)
            
            # Use the specific model trained for this step
            model_step = step + 1  # Models are 1-indexed
            
            if model_step not in self.models:
                print(f"DEBUG: WARNING - Model for step {model_step} not found!")
                continue
                
            model = self.models[model_step]
            
            # Make prediction using the same input features for all steps
            pred_value = model.predict(X_pred)[0]
            
            # Inverse transform if log was applied
            if self.log_transform:
                pred_value = np.expm1(pred_value)
            
            # Ensure non-negative
            pred_value = max(0, pred_value)
            predictions.append(pred_value)
            
            print(f"DEBUG: Step {step+1}: date={next_date.date()}, prediction={pred_value:.0f}")
        
        print(f"DEBUG: Final predictions count: {len(predictions)}")
        print(f"DEBUG: Final dates count: {len(forecast_dates)}")
        
        # If horizon > prediction_length, use recursive for remaining steps
        if horizon > self.prediction_length:
            print(f"Warning: Direct forecasting limited to {self.prediction_length} steps. Using recursive for remaining {horizon - self.prediction_length} steps.")
            
            # Get the last prediction date and continue recursively
            last_date = forecast_dates[-1]
            remaining_horizon = horizon - self.prediction_length
            
            # Create current state for recursive continuation
            current_data = hist_data.copy()
            
            # Add the direct predictions to current_data
            for i, pred_val in enumerate(predictions):
                pred_date = start_date + timedelta(days=date_diff * i)
                pred_row = pd.DataFrame({
                    self.date_col: [pred_date],
                    self.country_col: [country],
                    self.target_col: [pred_val if not self.log_transform else np.log1p(max(0, pred_val))]
                })
                current_data = pd.concat([current_data, pred_row], ignore_index=True)
            
            # Continue with recursive forecasting for remaining steps
            recursive_start = last_date + timedelta(days=date_diff)
            recursive_result = self._predict_recursive(country, recursive_start, remaining_horizon, current_data.tail(self.context_length))
            
            # Combine results
            forecast_dates.extend(recursive_result['date'].tolist())
            predictions.extend(recursive_result['forecast'].tolist())
        
        # Return results in the same format as TransformerForecastModel
        result_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': predictions
        })
        
        print(f"DEBUG: Returning result with shape: {result_df.shape}")
        return result_df

    def get_feature_importance(self) -> dict:
        """Get feature importance from trained models"""
        if not self.models:
            raise ValueError("Model not trained. Call train() first.")
            
        # Average feature importance across all prediction steps
        importance_dict = {}
        for feature in self._feature_names:
            importance_values = [self.models[step].feature_importances_[self._feature_names.index(feature)] 
                               for step in self.models.keys()]
            importance_dict[feature] = np.mean(importance_values)
            
        return importance_dict 