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
    """
    Improved Transformer for weekly time series with enhanced features and error handling.
    """
    def __init__(
        self,
        countries: list[str],
        date_col: str = "Date_reported",
        country_col: str = "Country",
        target_col: str = "New_cases",
        context_length: int = 52,
        prediction_length: int = 8,
        lags_sequence: list[int] = None,
        distribution_output: str = "negative_binomial",
        embedding_dim: int = 64,
        num_parallel_samples: int = 200,
        epochs: int = 25,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.05,
        batch_size: int = 32,
        log_transform: bool = False,
        clip_negative: bool = True,
        device: str = None,
        num_time_features: int = 8,
        d_model: int = 256,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        encoder_attention_heads: int = 8,
        decoder_attention_heads: int = 8,
        encoder_ffn_dim: int = 512,
        decoder_ffn_dim: int = 512,
        dropout: float = 0.2,
        use_cache: bool = True
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
        self.num_time_features = num_time_features

        # Validate distribution and transform settings
        if distribution_output == "negative_binomial" and log_transform:
            print("Warning: negative_binomial requires integers. Disabling log_transform.")
            self.log_transform = False

        self.distribution_output = distribution_output

        # Default lags sequence if none provided
        if lags_sequence is None:
            lags_sequence = [1, 2, 3, 4, 8, 12, 24, 52]

        # Filter out lags that are too long
        self.lags_sequence = [lag for lag in lags_sequence if lag < context_length]
        if len(self.lags_sequence) < len(lags_sequence):
            dropped = set(lags_sequence) - set(self.lags_sequence)
            print(f"Warning: dropping lags >= context_length: {dropped}")

        self.max_lag = max(self.lags_sequence, default=0)
        self.history_length = context_length + self.max_lag

        # Custom time features function
        self._custom_time_features_fn = None

        # Model configuration with enhanced architecture
        config = TimeSeriesTransformerConfig(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            lags_sequence=self.lags_sequence,
            distribution_output=self.distribution_output,
            input_size=1,
            scaling=None,  # Use external scaling
            num_time_features=self.num_time_features,
            num_dynamic_real_features=0,
            num_static_categorical_features=1,
            num_static_real_features=0,
            cardinality=[self._num_countries],
            embedding_dimension=[embedding_dim],
            num_parallel_samples=num_parallel_samples,
            # Enhanced architecture
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            encoder_attention_heads=encoder_attention_heads,
            decoder_attention_heads=decoder_attention_heads,
            encoder_ffn_dim=encoder_ffn_dim,
            decoder_ffn_dim=decoder_ffn_dim,
            d_model=d_model,
            dropout=dropout,
            attention_dropout=dropout,
            activation_dropout=dropout,
            encoder_layerdrop=0.1,
            decoder_layerdrop=0.1,
            activation_function='gelu',
            init_std=0.02,
            use_cache=use_cache,
        )

        self.model = TimeSeriesTransformerForPrediction(config).to(self.device)
        self._train_df = None

        print(f"Model initialized on {self.device}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def set_time_features_fn(self, fn):
        """Set custom time features function"""
        self._custom_time_features_fn = fn

    def make_time_features(self, dates) -> np.ndarray:
        """Generate time features with custom function support"""
        if self._custom_time_features_fn is not None:
            return self._custom_time_features_fn(dates)
        else:
            # Default implementation with 8 features
            dates = pd.DatetimeIndex(dates)

            # Basic features
            week_of_year = dates.isocalendar().week.to_numpy() / 52
            month = dates.month.to_numpy() / 12
            quarter = dates.quarter.to_numpy() / 4

            # Cyclical encoding
            month_sin = np.sin(2 * np.pi * dates.month / 12)
            month_cos = np.cos(2 * np.pi * dates.month / 12)
            week_sin = np.sin(2 * np.pi * dates.isocalendar().week / 52)
            week_cos = np.cos(2 * np.pi * dates.isocalendar().week / 52)

            # Seasonal indicator
            is_winter = ((dates.month >= 11) | (dates.month <= 2)).astype(float)

            return np.stack([
                week_of_year, month, quarter,
                month_sin, month_cos, week_sin, week_cos,
                is_winter
            ], axis=1)

    def preprocess_values(self, country, values):
        """Preprocess values based on distribution type"""
        values = np.array(values, dtype=np.float32)

        if self.distribution_output == "negative_binomial":
            # Ensure non-negative integers
            values = np.maximum(0, values)

            # Handle outliers with adaptive threshold
            if len(values) > 10:
                # Use robust statistics
                Q1 = np.percentile(values[values > 0], 25) if np.any(values > 0) else 0
                Q3 = np.percentile(values[values > 0], 75) if np.any(values > 0) else 1
                IQR = Q3 - Q1

                # Less aggressive capping for negative binomial
                upper_bound = Q3 + 3 * IQR
                values = np.clip(values, 0, upper_bound)

            # Round to integers
            return np.round(values).astype(np.float32)

        elif self.distribution_output == "student_t":
            if self.log_transform:
                # Log transform with small constant
                return np.log1p(np.maximum(0, values))
            else:
                # Clip extreme values
                if len(values) > 10:
                    upper_percentile = np.percentile(values, 99.5)
                    values = np.clip(values, 0, upper_percentile)
                return values

        else:
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
        """Train the model with improved error handling"""
        # Ensure date column is datetime
        train_data = train_data.copy()
        train_data[self.date_col] = pd.to_datetime(train_data[self.date_col])

        examples = {
            "past_values": [],
            "future_values": [],
            "static_categorical_features": [],
            "past_time_features": [],
            "future_time_features": []
        }

        frames = []

        for country in self.countries:
            dfc = train_data[train_data[self.country_col] == country].copy()

            if len(dfc) == 0:
                print(f"Warning: No data for country {country}")
                continue

            dfc = dfc.set_index(self.date_col)

            # Ensure weekly frequency
            dfc = dfc.asfreq('W', method='pad').reset_index()
            frames.append(dfc.assign(**{self.country_col: country}))

            dates = dfc[self.date_col]
            raw_vals = dfc[self.target_col].to_numpy()

            # Preprocess values
            vals = self.preprocess_values(country, raw_vals)

            # Create training examples
            n = len(vals) - (self.history_length + self.prediction_length) + 1

            if n <= 0:
                print(f"Warning: Insufficient data for {country} (need {self.history_length + self.prediction_length} points, have {len(vals)})")
                continue

            for i in range(n):
                # Extract windows
                hist = vals[i:i+self.history_length]
                fut = vals[i+self.history_length:i+self.history_length+self.prediction_length]
                hist_dates = dates.iloc[i:i+self.history_length]

                # Generate future dates
                last_hist_date = hist_dates.iloc[-1]
                fut_dates = pd.date_range(
                    start=last_hist_date + pd.Timedelta(weeks=1),
                    periods=self.prediction_length,
                    freq='W'
                )

                # Create features
                past_features = self.make_time_features(hist_dates)
                future_features = self.make_time_features(fut_dates)

                # Validate shapes
                assert past_features.shape == (self.history_length, self.num_time_features)
                assert future_features.shape == (self.prediction_length, self.num_time_features)

                # Add to examples
                examples["past_values"].append(hist.tolist())
                examples["future_values"].append(fut.tolist())
                examples["static_categorical_features"].append(self._country_to_id[country])
                examples["past_time_features"].append(past_features.tolist())
                examples["future_time_features"].append(future_features.tolist())

        if not examples["past_values"]:
            raise ValueError("No valid training examples created")

        self._train_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        # Create dataset
        train_ds = Dataset.from_dict(examples)

        print(f"Created {len(train_ds)} training examples")

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./transformer_output',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            logging_steps=50,
            save_steps=100,
            remove_unused_columns=False,
            dataloader_drop_last=True,
            report_to=[],
            label_names=["future_values"],
        )

        # Collate function
        def collate_fn(batch):
            return {
                'past_values': torch.tensor(
                    [b['past_values'] for b in batch],
                    dtype=torch.float32
                ).to(self.device),
                'past_time_features': torch.tensor(
                    [b['past_time_features'] for b in batch],
                    dtype=torch.float32
                ).to(self.device),
                'past_observed_mask': torch.ones(
                    (len(batch), self.history_length),
                    dtype=torch.bool
                ).to(self.device),
                'future_values': torch.tensor(
                    [b['future_values'] for b in batch],
                    dtype=torch.float32
                ).to(self.device),
                'future_time_features': torch.tensor(
                    [b['future_time_features'] for b in batch],
                    dtype=torch.float32
                ).to(self.device),
                'future_observed_mask': torch.ones(
                    (len(batch), self.prediction_length),
                    dtype=torch.bool
                ).to(self.device),
                'static_categorical_features': torch.tensor(
                    [[b['static_categorical_features']] for b in batch],
                    dtype=torch.long
                ).to(self.device)
            }

        # Train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=collate_fn
        )

        print("Starting training...")
        train_result = trainer.train()
        print(f"Training completed. Final loss: {train_result.training_loss:.4f}")

        # Clear GPU cache
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        # Get date statistics
        dates_all = pd.to_datetime(self._train_df[self.date_col]).sort_values()

        return {
            'min_date': dates_all.min(),
            'max_date': dates_all.max(),
            'training_loss': train_result.training_loss,
            'num_examples': len(train_ds)
        }

    def predict(self, country: str, start_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
        """Generate predictions with improved stability"""
        # Ensure start_date is pandas Timestamp
        start_date = pd.to_datetime(start_date)

        # Get historical data
        dfc = self._train_df[self._train_df[self.country_col] == country].sort_values(self.date_col)
        dfc[self.date_col] = pd.to_datetime(dfc[self.date_col])

        # Find data before start_date
        history_data = dfc[dfc[self.date_col] <= start_date].sort_values(self.date_col)

        if len(history_data) < self.history_length:
            raise ValueError(
                f"Insufficient history for {country} before {start_date}. "
                f"Need {self.history_length} points, have {len(history_data)}"
            )

        # Get most recent history_length points
        history_data = history_data.iloc[-self.history_length:]
        hist_dates = history_data[self.date_col]
        raw_vals = history_data[self.target_col].to_numpy()

        # Get recent statistics for post-processing
        recent_window = history_data.tail(8)
        recent_mean = recent_window[self.target_col].mean()
        recent_std = recent_window[self.target_col].std() + 1e-8
        recent_max = recent_window[self.target_col].max()

        # Preprocess values
        vals = self.preprocess_values(country, raw_vals)

        # Generate future dates
        last_hist_date = hist_dates.iloc[-1]
        fut_dates = pd.date_range(
            start=last_hist_date + pd.Timedelta(weeks=1),
            periods=horizon,
            freq='W'
        )

        # Create model inputs
        past = torch.tensor(vals, dtype=torch.float32).unsqueeze(0).to(self.device)
        past_features = self.make_time_features(hist_dates)
        future_features = self.make_time_features(fut_dates)

        ptf = torch.tensor(past_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        ftf = torch.tensor(future_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        sc = torch.tensor([[self._country_to_id[country]]], dtype=torch.long).to(self.device)

        # Generate predictions
        with torch.no_grad():
            outputs = self.model.generate(
                past_values=past,
                past_time_features=ptf,
                future_time_features=ftf,
                static_categorical_features=sc,
                # num_parallel_samples=self.model.config.num_parallel_samples
            )

        # Extract predictions
        predictions = outputs.sequences.cpu().numpy()[0]  # Shape: (num_samples, horizon)

        # Process predictions
        processed_predictions = []
        for sample in predictions:
            processed = self.postprocess_values(country, sample)
            processed_predictions.append(processed)

        predictions_array = np.array(processed_predictions)

        # Calculate robust statistics
        median_forecast = np.median(predictions_array, axis=0)
        mean_forecast = np.mean(predictions_array, axis=0)
        lower_10 = np.percentile(predictions_array, 10, axis=0)
        upper_90 = np.percentile(predictions_array, 90, axis=0)

        # Use median as primary forecast (more robust)
        forecast = median_forecast

        # Apply reality checks
        for i in range(len(forecast)):
            # Gradual trend adjustment
            if i == 0:
                # First prediction shouldn't deviate too much from recent history
                max_change = recent_std * 3
                if abs(forecast[i] - recent_mean) > max_change:
                    forecast[i] = recent_mean + np.sign(forecast[i] - recent_mean) * max_change
            else:
                # Subsequent predictions should have reasonable growth
                max_growth = 2.0  # Maximum 2x growth per week
                if forecast[i] > forecast[i-1] * max_growth:
                    forecast[i] = forecast[i-1] * max_growth

            # Cap at reasonable multiple of recent maximum
            forecast[i] = min(forecast[i], recent_max * 5)

        # Create output DataFrame
        prediction_df = pd.DataFrame({
            'date': fut_dates,
            'forecast': forecast,
            'forecast_median': median_forecast,
            'forecast_mean': mean_forecast,
            'lower_10': lower_10,
            'lower_90': lower_10,  # This seems like it should be renamed
            'upper_90': upper_90,
            'uncertainty': upper_90 - lower_10
        })

        return prediction_df

    def evaluate_and_plot(self, country: str, start_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
        """Evaluate predictions and create comprehensive plot"""
        start_date = pd.to_datetime(start_date)

        # Generate predictions
        pred = self.predict(country, start_date, horizon)
        pred['date'] = pd.to_datetime(pred['date'])

        # Get historical data for context
        history_weeks = 26
        history_end_date = start_date
        history_start_date = history_end_date - pd.Timedelta(weeks=history_weeks)

        historical_data = self._train_df[
            (self._train_df[self.country_col] == country) &
            (pd.to_datetime(self._train_df[self.date_col]) >= history_start_date) &
            (pd.to_datetime(self._train_df[self.date_col]) <= history_end_date)
        ].copy()

        historical_data = historical_data.rename(columns={self.date_col: 'date', self.target_col: 'actual'})
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        historical_data = historical_data.sort_values('date')

        # Get actual values for evaluation
        actual_forecast_period = self._train_df[
            (self._train_df[self.country_col] == country) &
            (pd.to_datetime(self._train_df[self.date_col]) > start_date) &
            (pd.to_datetime(self._train_df[self.date_col]) <= start_date + pd.Timedelta(weeks=horizon))
        ].copy()

        actual_forecast_period = actual_forecast_period.rename(columns={self.date_col: 'date', self.target_col: 'actual'})
        actual_forecast_period['date'] = pd.to_datetime(actual_forecast_period['date'])
        actual_forecast_period = actual_forecast_period.sort_values('date')

        # Merge for evaluation
        comp = pd.merge(actual_forecast_period, pred, on='date', how='inner')

        # Create comprehensive plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Main forecast plot
        ax1.plot(historical_data['date'], historical_data['actual'],
                 marker='o', color='blue', linestyle='-', linewidth=2,
                 label='Historical Data', markersize=4)

        if not actual_forecast_period.empty:
            ax1.plot(actual_forecast_period['date'], actual_forecast_period['actual'],
                     marker='o', color='green', linestyle='-', linewidth=2,
                     label='Actual (Test Period)', markersize=6)

        ax1.plot(pred['date'], pred['forecast'],
                 marker='s', color='red', linestyle='--', linewidth=2,
                 label='Forecast', markersize=6)

        # Prediction intervals
        ax1.fill_between(pred['date'], pred['lower_90'], pred['upper_90'],
                         color='red', alpha=0.2, label='90% Prediction Interval')

        ax1.axvline(x=start_date, color='black', linestyle='--', alpha=0.7,
                    label='Forecast Start')

        ax1.set_title(f'{country}: COVID-19 New Cases - Forecast vs Actual', fontsize=16)
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('New Cases', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('symlog')  # Symmetric log scale

        # Error analysis plot (if we have actuals)
        if not comp.empty:
            errors = comp['actual'] - comp['forecast']
            ax2.bar(comp['date'], errors, color='darkblue', alpha=0.7, label='Forecast Error')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_title('Forecast Errors (Actual - Forecast)', fontsize=14)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Error', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # Calculate metrics
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors**2))
            mape = np.mean(np.abs(errors / (comp['actual'] + 1))) * 100

            metrics_text = f'MAE: {mae:,.0f} | RMSE: {rmse:,.0f} | MAPE: {mape:.1f}%'
            ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                     verticalalignment='top', fontsize=12)

        plt.tight_layout()
        plt.show()

        return comp if not comp.empty else pred