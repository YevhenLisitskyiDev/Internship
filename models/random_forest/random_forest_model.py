import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error
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