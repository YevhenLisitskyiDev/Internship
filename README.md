# Time Series Forecasting Pipelines

This project provides a unified interface to run multiple time series forecasting pipelines (Random Forest, Transformer, XGBoost) on COVID-19 data.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Main Menu

```bash
python main.py
```

You will be prompted to select a pipeline:
- **Random Forest Pipeline**
- **Transformer Pipeline**
- **XGBoost Pipeline**

Each pipeline will run cross-validation on the Italy COVID-19 time series and output results to the `outputs/` directory.

## Project Structure

- `main.py` — Entry point. Use this to select and run pipelines.
- `pipelines/` — Contains all actual pipeline code:
    - `random_forest/`
    - `transformer/`
    - `xgboost/`
- `models/` — Model definitions for each pipeline.
- `data/` — Place your input CSV here.
- `outputs/` — Results and predictions are saved here.

**Note:**
> All files outside the folders (except `main.py` and `requirements.txt`) are redundant and only kept to ensure everything works after refactoring. You can ignore or delete them.

## Customization
- To change the country or experiment settings, edit the relevant pipeline script in `pipelines/`.

## Requirements
See `requirements.txt` for all dependencies.

## Data Versioning

- By default, **all CSV files are ignored** by git (see `.gitignore`).
- The main data file `data/WHO-COVID-19-global-data.csv` is explicitly tracked and will be included in commits.
- If you want to track additional CSV files in `data/` or elsewhere, you must update `.gitignore` to allow them.

---

For any issues, please check the code comments or pipeline scripts for further details. 