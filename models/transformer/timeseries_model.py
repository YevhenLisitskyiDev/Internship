import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from transformers import (
    TimeSeriesTransformerConfig,
    TimeSeriesTransformerForPrediction,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import warnings
warnings.filterwarnings('ignore')


class TransformerForecastModel:

    def __init__(
        self,
        countries: list[str],
        date_col: str = "Date_reported",
        country_col: str = "Country",
        target_col: str = "New_cases",
        context_length: int = 64,  # FIXED: Increased from 52 to 64 (larger than max lag 52)
        prediction_length: int = 8,
        lags_sequence: list[int] = None,
        distribution_output: str = "student_t",  # Default from docs
        loss: str = "nll",  # Negative log likelihood - optimal for probabilistic forecasting
        embedding_dim: int = 32,  # Reduced from 64
        num_parallel_samples: int = 100,  # Default from docs
        epochs: int = 25,
        learning_rate: float = 1e-4,  # Better default
        weight_decay: float = 0.01,  # Reduced
        batch_size: int = 32,
        log_transform: bool = False,
        clip_negative: bool = True,
        device: str = None,
        # Enhanced architecture based on docs
        d_model: int = 128,  # Increased from 64 default
        encoder_layers: int = 4,  # Increased from 2 default
        decoder_layers: int = 4,  # Increased from 2 default
        encoder_attention_heads: int = 8,  # Increased from 2 default  
        decoder_attention_heads: int = 8,  # Increased from 2 default
        encoder_ffn_dim: int = 256,  # Increased from 32 default
        decoder_ffn_dim: int = 256,  # Increased from 32 default
        dropout: float = 0.1,  # Default from docs
        attention_dropout: float = 0.1,  # Added
        activation_dropout: float = 0.1,  # Added
        encoder_layerdrop: float = 0.0,  # Conservative default
        decoder_layerdrop: float = 0.0,  # Conservative default
        scaling: str = "mean",  # Enable proper scaling
        init_std: float = 0.02,  # Default from docs
        use_cache: bool = True,
        num_time_features: int = 12
    ):
        self.countries = countries
        self.date_col = date_col
        self.country_col = country_col
        self.target_col = target_col
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.log_transform = log_transform
        self.clip_negative = clip_negative
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._country_to_id = {c: i for i, c in enumerate(countries)}
        self._num_countries = len(countries)

        # Validate distribution and transform settings
        if distribution_output == "negative_binomial" and log_transform:
            print("Warning: negative_binomial requires integers. Disabling log_transform.")
            self.log_transform = False

        self.distribution_output = distribution_output
        self.loss = loss
        self.scaling = scaling

        # Enhanced lags sequence based on docs examples
        if lags_sequence is None:
            # More comprehensive lags for weekly data
            lags_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 12, 24, 52]

        # Filter out lags that are too long
        self.lags_sequence = [lag for lag in lags_sequence if lag < context_length]
        if len(self.lags_sequence) < len(lags_sequence):
            dropped = set(lags_sequence) - set(self.lags_sequence)
            print(f"Warning: dropping lags >= context_length: {dropped}")

        self.max_lag = max(self.lags_sequence, default=0)
        self.history_length = context_length + self.max_lag

        # Enhanced time features (increased from 8 to 12)
        self.num_time_features = num_time_features

        # Model configuration with enhanced architecture based on HF docs
        config = TimeSeriesTransformerConfig(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            distribution_output=self.distribution_output,
            loss=self.loss,  # Explicit loss specification
            input_size=1,
            lags_sequence=self.lags_sequence,
            scaling=self.scaling,  # Enable built-in scaling
            num_time_features=self.num_time_features,
            num_dynamic_real_features=0,
            num_static_categorical_features=1,
            num_static_real_features=0,
            cardinality=[self._num_countries],
            embedding_dimension=[embedding_dim],
            num_parallel_samples=num_parallel_samples,
            # Enhanced architecture parameters
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            decoder_attention_heads=decoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            decoder_ffn_dim=decoder_ffn_dim,
            d_model=d_model,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            encoder_layerdrop=encoder_layerdrop,
            decoder_layerdrop=decoder_layerdrop,
            activation_function='gelu',  # Explicit activation
            init_std=init_std,
            use_cache=use_cache,
            is_encoder_decoder=True,  # Explicit specification
        )

        self.model = TimeSeriesTransformerForPrediction(config).to(self.device)
        self._train_df = None

        print(f"Enhanced model initialized on {self.device}")
        print(f"Distribution: {self.distribution_output}, Loss: {self.loss}, Scaling: {self.scaling}")
        print(f"Architecture: {d_model}d, {encoder_layers}E/{decoder_layers}D layers, {encoder_attention_heads} heads")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def make_time_features(self, dates) -> np.ndarray:
        """Enhanced time features with 12 components based on docs best practices"""
        dates = pd.DatetimeIndex(dates)

        # Basic temporal features
        day_of_week = dates.dayofweek.to_numpy() / 6  # 0-6 normalized
        week_of_year = dates.isocalendar().week.to_numpy() / 52  # 1-52 normalized
        month = dates.month.to_numpy() / 12  # 1-12 normalized
        quarter = dates.quarter.to_numpy() / 4  # 1-4 normalized
        day_of_month = dates.day.to_numpy() / 31  # 1-31 normalized

        # Cyclical encoding (better for periodicity)
        month_sin = np.sin(2 * np.pi * dates.month / 12)
        month_cos = np.cos(2 * np.pi * dates.month / 12)
        week_sin = np.sin(2 * np.pi * dates.isocalendar().week / 52)
        week_cos = np.cos(2 * np.pi * dates.isocalendar().week / 52)

        # Seasonal indicators
        is_winter = ((dates.month >= 11) | (dates.month <= 2)).astype(float)
        is_summer = ((dates.month >= 5) & (dates.month <= 8)).astype(float)
        
        # Holiday proximity (simplified)
        is_year_end = ((dates.month == 12) & (dates.day >= 20)).astype(float)

        return np.stack([
            day_of_week, week_of_year, month, quarter, day_of_month,
            month_sin, month_cos, week_sin, week_cos,
            is_winter, is_summer, is_year_end
        ], axis=1)

    def preprocess_values(self, country, values):
        """Enhanced preprocessing with proper handling for different distributions"""
        values = np.array(values, dtype=np.float32)

        if self.distribution_output == "negative_binomial":
            # Ensure non-negative integers with better outlier handling
            values = np.maximum(0, values)

            # More robust outlier detection
            if len(values) > 20:
                # Use IQR method but adapted for count data
                non_zero_values = values[values > 0]
                if len(non_zero_values) > 10:
                    Q1 = np.percentile(non_zero_values, 25)
                    Q3 = np.percentile(non_zero_values, 75)
                    IQR = Q3 - Q1
                    # More conservative bound for negative binomial
                    upper_bound = Q3 + 2.5 * IQR
                    values = np.clip(values, 0, upper_bound)

            return np.round(values).astype(np.float32)

        elif self.distribution_output == "student_t":
            if self.log_transform:
                # Stable log transform
                return np.log1p(np.maximum(0, values))
            else:
                # Robust scaling for student_t
                if len(values) > 10:
                    # Use robust percentiles
                    lower_bound = np.percentile(values, 1)
                    upper_bound = np.percentile(values, 99)
                    values = np.clip(values, lower_bound, upper_bound)
                return values

        elif self.distribution_output == "normal":
            # For normal distribution, handle outliers more conservatively
            if len(values) > 10:
                mean_val = np.mean(values)
                std_val = np.std(values)
                # 3-sigma rule
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                values = np.clip(values, lower_bound, upper_bound)
            return values

        return values

    def postprocess_values(self, country, values):
        """Inverse preprocessing transformations"""
        values = np.array(values)

        if self.log_transform and self.distribution_output == "student_t":
            # Safe expm1 to prevent overflow
            with np.errstate(over='raise'):
                try:
                    values = np.expm1(np.clip(values, -10, 10))
                except FloatingPointError:
                    values = np.exp(np.clip(values, -10, 10)) - 1

        if self.clip_negative:
            values = np.maximum(0, values)

        if self.distribution_output == "negative_binomial":
            values = np.round(values)

        return values

    def train(self, train_data: pd.DataFrame) -> dict:
        """Train transformer model on expanding window data (simplified approach)"""
        # Store training data for later use in predict method
        self._train_df = train_data.copy()
        
        print("Training Transformer model...")
        
        # For expanding window CV, we don't need complex sliding windows
        # Just train the model to understand the pattern in the data
        # The prediction will use the most recent context_length portion
        
        # Prepare the data for the country
        country_data = []
        for country in self.countries:
            dfc = train_data[train_data[self.country_col] == country].sort_values(self.date_col)
            dfc[self.date_col] = pd.to_datetime(dfc[self.date_col])
            
            # FIXED: Need enough data for context + max_lag + prediction
            min_required = self.context_length + max(self.lags_sequence) + self.prediction_length
            if len(dfc) < min_required:
                print(f"Warning: {country} has insufficient data ({len(dfc)} < {min_required})")
                continue

            # Get enough recent data for lags + context + prediction
            # HuggingFace model needs: past_length = context_length + max(lags)
            past_length = self.context_length + max(self.lags_sequence)
            total_needed = past_length + self.prediction_length
            recent_data = dfc.tail(total_needed)
            
            # Past values: enough for context + lags
            past_values = recent_data[self.target_col].values[:past_length]
            # Future values: what to predict
            future_values = recent_data[self.target_col].values[past_length:past_length + self.prediction_length]
            
            # Create time features for the full sequence
            dates = pd.date_range(
                start=recent_data.iloc[0][self.date_col],
                periods=len(recent_data),
                freq='W-SUN'
                )

            # Time features for past (context) and future (prediction)
            past_time_features = self.make_time_features(dates[:past_length])
            future_time_features = self.make_time_features(dates[past_length:past_length + self.prediction_length])
            
            print(f"Created training example for {country}: past_length={len(past_values)}, future_length={len(future_values)}")
            
            # Store this training example
            country_data.append({
                'past_values': past_values.tolist(),
                'future_values': future_values.tolist(),
                'static_categorical_features': [self._country_to_id[country]],
                'past_time_features': past_time_features.tolist(),
                'future_time_features': future_time_features.tolist(),
                'past_observed_mask': [True] * past_length,
                'future_observed_mask': [True] * self.prediction_length
            })
        
        if not country_data:
            raise ValueError("No valid training data created!")

        print(f"✅ Created training data for {len(country_data)} countries")

        # Convert to HuggingFace Dataset
        train_ds = Dataset.from_dict({
            key: [item[key] for item in country_data]
            for key in country_data[0].keys()
        })
        
        # Simple training arguments
        training_args = TrainingArguments(
            output_dir='./transformer_output',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=1,  # Small batch for simplicity
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            logging_steps=1,
            save_steps=999999,
            eval_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )

        def collate_fn(batch):
            batch_dict = {}
            for key in batch[0].keys():
                if key == "static_categorical_features":
                    batch_dict[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
                elif "mask" in key:
                    batch_dict[key] = torch.tensor([item[key] for item in batch], dtype=torch.bool)
                else:
                    batch_dict[key] = torch.tensor([item[key] for item in batch], dtype=torch.float32)
            return batch_dict
        
        print(f"🚀 Starting training...")
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=collate_fn,
        )

        train_result = trainer.train()
        print(f"✅ Training completed. Final loss: {train_result.training_loss:.4f}")

        return {
            'training_loss': train_result.training_loss,
            'num_examples': len(train_ds),
            'distribution': self.distribution_output,
            'loss_type': self.loss,
            'scaling': self.scaling
        }

    def predict(self, country: str, start_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
        """Enhanced prediction using model.generate() method from HF docs"""
        # Ensure start_date is pandas Timestamp
        start_date = pd.to_datetime(start_date)

        # Get historical data
        dfc = self._train_df[self._train_df[self.country_col] == country].sort_values(self.date_col)
        dfc[self.date_col] = pd.to_datetime(dfc[self.date_col])

        # Find data before start_date
        history_data = dfc[dfc[self.date_col] <= start_date].sort_values(self.date_col)

        # FIXED: Need enough data for context + max_lag (same as training)
        past_length = self.context_length + max(self.lags_sequence)
        if len(history_data) < past_length:
            raise ValueError(
                f"Insufficient history for {country} before {start_date}. "
                f"Need {past_length} points, have {len(history_data)}"
            )

        # Get most recent past_length points (enough for context + lags)
        history_data = history_data.iloc[-past_length:]
        hist_dates = history_data[self.date_col]
        raw_vals = history_data[self.target_col].to_numpy()

        # Preprocess historical values
        hist_vals = self.preprocess_values(country, raw_vals)
        
        # Create observed mask for historical data
        hist_observed = ~pd.isna(raw_vals)
        hist_vals = np.nan_to_num(hist_vals, nan=0.0)

        # Generate future dates
        future_dates = pd.date_range(
            start=start_date + pd.Timedelta(weeks=1),
            periods=horizon,
            freq='W-SUN'
        )

        # Create time features
        past_time_features = self.make_time_features(hist_dates)
        future_time_features = self.make_time_features(future_dates)

        # Prepare inputs for generation (following HF docs format)
        past_values_tensor = torch.tensor(
            hist_vals, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        past_time_features_tensor = torch.tensor(
            past_time_features, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        past_observed_mask_tensor = torch.tensor(
            hist_observed, dtype=torch.bool
        ).unsqueeze(0).to(self.device)
        static_cat_tensor = torch.tensor(
            [[self._country_to_id[country]]], dtype=torch.long
        ).to(self.device)
        future_time_features_tensor = torch.tensor(
            future_time_features, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        # Enhanced prediction using model.generate() method from HF docs
        try:
            # Generate predictions using documented parameters only
            outputs = self.model.generate(
                past_values=past_values_tensor,
                past_time_features=past_time_features_tensor,
                past_observed_mask=past_observed_mask_tensor,
                static_categorical_features=static_cat_tensor,
                future_time_features=future_time_features_tensor
            )
            
            # Extract prediction sequences
            prediction_outputs = outputs.sequences  # Shape: (batch, num_parallel_samples, prediction_length)
            
            # Get mean prediction (average across samples)
            predictions = prediction_outputs.mean(dim=1).cpu().numpy()  # Shape: (batch, prediction_length)
            
        except Exception as e:
            print(f"Warning: generate() failed ({e}), falling back to forward pass")
            # Fallback to forward pass if generate fails
            with torch.no_grad():
                outputs = self.model(
                    past_values=past_values_tensor,
                    past_time_features=past_time_features_tensor,
                    past_observed_mask=past_observed_mask_tensor,
                    static_categorical_features=static_cat_tensor,
                    future_time_features=future_time_features_tensor
                )
                predictions = outputs.prediction_outputs.mean(dim=-1).cpu().numpy()

        # Handle different horizon lengths
        if horizon <= self.prediction_length:
            # Direct prediction
            final_predictions = predictions[:, :horizon]
        else:
            # Multi-step generation for longer horizons
            print(f"Long horizon ({horizon} > {self.prediction_length}), using iterative prediction")
            final_predictions = []
            
            current_horizon = min(horizon, self.prediction_length)
            remaining_horizon = horizon - current_horizon
            
            # First prediction
            current_pred = predictions[:, :current_horizon]
            final_predictions.append(current_pred)
            
            # Iterative prediction for remaining horizon
            current_history = hist_vals.copy()
            current_dates = hist_dates.tolist()
            
            while remaining_horizon > 0:
                # Update history with previous predictions (using median)
                median_pred = np.median(current_pred, axis=0)
                current_history = np.concatenate([current_history[len(median_pred):], median_pred])
                
                # Update dates
                last_date = pd.to_datetime(current_dates[-1])
                new_dates = pd.date_range(
                    start=last_date + pd.Timedelta(weeks=1),
                    periods=len(median_pred),
                    freq='W-SUN'
                )
                current_dates = current_dates[len(median_pred):] + new_dates.tolist()
                
                # Prepare for next prediction
                next_horizon = min(remaining_horizon, self.prediction_length)
                next_future_dates = pd.date_range(
                    start=pd.to_datetime(current_dates[-1]) + pd.Timedelta(weeks=1),
                    periods=next_horizon,
                    freq='W-SUN'
                )
                
                # Create new inputs
                next_inputs = {
                    'past_values': torch.tensor(
                        current_history, dtype=torch.float32
                    ).unsqueeze(0).to(self.device),
                    'past_time_features': torch.tensor(
                        self.make_time_features(current_dates), dtype=torch.float32
                    ).unsqueeze(0).to(self.device),
                    'past_observed_mask': torch.ones(
                        (1, len(current_history)), dtype=torch.bool
                    ).to(self.device),
                    'future_time_features': torch.tensor(
                        self.make_time_features(next_future_dates), dtype=torch.float32
                    ).unsqueeze(0).to(self.device),
                    'static_categorical_features': torch.tensor(
                        [[self._country_to_id[country]]], dtype=torch.long
                    ).to(self.device)
                }
                
                # Generate next prediction
                with torch.no_grad():
                    next_outputs = self.model.generate(**next_inputs)
                    current_pred = next_outputs.sequences.cpu().numpy()[0, :, :next_horizon]
                
                final_predictions.append(current_pred)
                remaining_horizon -= next_horizon
            
            # Concatenate all predictions
            final_predictions = np.concatenate(final_predictions, axis=1)

        # Post-process predictions
        processed_predictions = []
        for i in range(final_predictions.shape[0]):
            pred = self.postprocess_values(country, final_predictions[i])
            processed_predictions.append(pred)
        
        processed_predictions = np.array(processed_predictions)

        # Calculate statistics
        mean_forecast = np.mean(processed_predictions, axis=0)
        median_forecast = np.median(processed_predictions, axis=0)
        std_forecast = np.std(processed_predictions, axis=0)

        # Quantiles for uncertainty intervals
        q10 = np.percentile(processed_predictions, 10, axis=0)
        q25 = np.percentile(processed_predictions, 25, axis=0)
        q75 = np.percentile(processed_predictions, 75, axis=0)
        q90 = np.percentile(processed_predictions, 90, axis=0)

        # Create result DataFrame
        result_df = pd.DataFrame({
            'Date_reported': future_dates[:horizon],
            'Country': country,
            'Forecast_mean': mean_forecast,
            'Forecast_median': median_forecast,
            'Forecast_std': std_forecast,
            'Forecast_lower_10': q10,
            'Forecast_lower_25': q25,
            'Forecast_upper_75': q75,
            'Forecast_upper_90': q90,
        })

        print(f"Generated {len(processed_predictions)} samples for {horizon}-week forecast")
        print(f"Forecast range: {mean_forecast.min():.1f} to {mean_forecast.max():.1f}")

        return result_df

    def evaluate_and_plot(self, country: str, start_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
        """Enhanced evaluation with better plotting and metrics"""
        forecast_df = self.predict(country, start_date, horizon)
        
        # Get actual data for comparison
        actual_data = self._train_df[
            (self._train_df[self.country_col] == country) &
            (pd.to_datetime(self._train_df[self.date_col]) > start_date)
        ].copy()

        if len(actual_data) == 0:
            print(f"Warning: No actual data available for {country} after {start_date}")
            return forecast_df
        
        actual_data[self.date_col] = pd.to_datetime(actual_data[self.date_col])
        actual_data = actual_data.sort_values(self.date_col)
        
        # Align forecast with actual data
        forecast_dates = pd.to_datetime(forecast_df['Date_reported'])
        actual_dates = pd.to_datetime(actual_data[self.date_col])
        
        # Find overlapping dates
        common_dates = forecast_dates[forecast_dates.isin(actual_dates)]
        
        if len(common_dates) == 0:
            print(f"Warning: No overlapping dates for evaluation")
            return forecast_df
        
        # Extract values for evaluation
        forecast_subset = forecast_df[forecast_df['Date_reported'].isin(common_dates)].copy()
        actual_subset = actual_data[actual_data[self.date_col].isin(common_dates)].copy()
        
        # Sort by date
        forecast_subset = forecast_subset.sort_values('Date_reported')
        actual_subset = actual_subset.sort_values(self.date_col)
        
        actual_values = actual_subset[self.target_col].values
        forecast_mean = forecast_subset['Forecast_mean'].values
        forecast_median = forecast_subset['Forecast_median'].values
        
        # Calculate enhanced metrics
        mae = np.mean(np.abs(actual_values - forecast_mean))
        mse = np.mean((actual_values - forecast_mean) ** 2)
        rmse = np.sqrt(mse)
        
        # MAPE with handling for zero values
        non_zero_mask = actual_values != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((actual_values[non_zero_mask] - forecast_mean[non_zero_mask]) / actual_values[non_zero_mask])) * 100
        else:
            mape = np.inf
        
        # Coverage metrics for uncertainty intervals
        lower_10 = forecast_subset['Forecast_lower_10'].values
        upper_90 = forecast_subset['Forecast_upper_90'].values
        lower_25 = forecast_subset['Forecast_lower_25'].values
        upper_75 = forecast_subset['Forecast_upper_75'].values
        
        coverage_80 = np.mean((actual_values >= lower_10) & (actual_values <= upper_90)) * 100
        coverage_50 = np.mean((actual_values >= lower_25) & (actual_values <= upper_75)) * 100
        
        # Directional accuracy
        if len(actual_values) > 1:
            actual_direction = np.diff(actual_values) > 0
            forecast_direction = np.diff(forecast_mean) > 0
            directional_accuracy = np.mean(actual_direction == forecast_direction) * 100
        else:
            directional_accuracy = np.nan
        
        print(f"\n📊 Enhanced Evaluation Results for {country}:")
        print(f"   Evaluation period: {len(common_dates)} weeks")
        print(f"   MAE: {mae:.2f}")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   80% Coverage: {coverage_80:.1f}%")
        print(f"   50% Coverage: {coverage_50:.1f}%")
        print(f"   Directional Accuracy: {directional_accuracy:.1f}%")
        
        # Enhanced plotting
        plt.figure(figsize=(14, 10))
        
        # Main plot
        plt.subplot(2, 1, 1)
        
        # Historical data (context)
        hist_data = self._train_df[
            (self._train_df[self.country_col] == country) &
            (pd.to_datetime(self._train_df[self.date_col]) <= start_date)
        ].copy()

        if len(hist_data) > 0:
            hist_data[self.date_col] = pd.to_datetime(hist_data[self.date_col])
            recent_hist = hist_data.tail(26)  # Last 6 months
            plt.plot(recent_hist[self.date_col], recent_hist[self.target_col], 
                    'b-', alpha=0.7, label='Historical', linewidth=1.5)

        # Actual values
        plt.plot(actual_subset[self.date_col], actual_values, 
                'g-', label='Actual', linewidth=2, marker='o', markersize=4)
        
        # Forecasts
        plt.plot(forecast_subset['Date_reported'], forecast_mean, 
                'r-', label='Forecast (Mean)', linewidth=2)
        plt.plot(forecast_subset['Date_reported'], forecast_median, 
                'orange', linestyle='--', label='Forecast (Median)', linewidth=1.5)
        
        # Uncertainty bands
        plt.fill_between(forecast_subset['Date_reported'], lower_10, upper_90,
                        alpha=0.2, color='red', label='80% Prediction Interval')
        plt.fill_between(forecast_subset['Date_reported'], lower_25, upper_75,
                        alpha=0.3, color='red', label='50% Prediction Interval')

        # Vertical line at forecast start
        plt.axvline(x=start_date, color='black', linestyle=':', alpha=0.7, label='Forecast Start')

        plt.title(f'Enhanced Time Series Forecast: {country}\n'
                 f'MAE: {mae:.1f}, RMSE: {rmse:.1f}, Coverage 80%: {coverage_80:.1f}%', 
                 fontsize=12, pad=20)
        plt.xlabel('Date')
        plt.ylabel(self.target_col)
        plt.legend(loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Residuals plot
        plt.subplot(2, 1, 2)
        residuals = actual_values - forecast_mean
        plt.plot(forecast_subset['Date_reported'], residuals, 'b-', marker='o', markersize=3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        plt.fill_between(forecast_subset['Date_reported'], 
                        -forecast_subset['Forecast_std'], forecast_subset['Forecast_std'],
                        alpha=0.2, color='gray', label='±1 Std')
        plt.title('Forecast Residuals')
        plt.xlabel('Date')
        plt.ylabel('Actual - Forecast')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        # Add evaluation metrics to the forecast DataFrame
        forecast_df_enhanced = forecast_df.copy()
        if len(common_dates) > 0:
            forecast_df_enhanced.attrs.update({
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'coverage_80': coverage_80,
                'coverage_50': coverage_50,
                'directional_accuracy': directional_accuracy,
                'evaluation_points': len(common_dates)
            })
        
        return forecast_df_enhanced