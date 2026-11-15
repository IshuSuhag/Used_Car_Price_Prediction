# Used Car Price Prediction

**Project:** Used Car Price Prediction Through Data Preprocessing, Exploratory Data Analysis, and Machine Learning

This beginner-friendly workflow uses `dataset/Car details v3.csv`, the richest CarDekho CSV in this project folder. Other available CSVs are `car data.csv`, `CAR DETAILS FROM CAR DEKHO.csv`, and `car details v4.csv`; their schemas differ. The source is the CarDekho Vehicle Dataset by Nehalbirla from Kaggle.

## Run

From the project folder, install dependencies and run:

```powershell
python -m pip install -r requirements.txt
python used_car_price_prediction.py
```

The reference year for vehicle age is 2026. The script uses a fixed random seed and 80:20 split. Missing feature values are imputed inside the training-fitted scikit-learn pipeline. Categorical features use one-hot encoding; numeric features use median imputation and standard scaling. The target selling price is never imputed. Exact duplicates are removed. Rows with unusable target, year outside 1900–2026, or missing/negative distance are excluded with counts recorded. IQR outliers are reported and charted, not automatically dropped, since they may be legitimate sales.

Feature engineering creates `brand` from the first token of `name` (a simple dataset-specific proxy) and `vehicle_age = 2026 - year`. Numeric portions of mileage, engine, and max_power are extracted from their unit strings; the original columns remain in the cleaned output. They are retained as model features where available.

## Outputs

All run products are written to `outputs/`: dataset inspection and cleaning notes, cleaned data, IQR outlier summary, EDA plots, model metric CSV and comparison plot, actual versus predicted prices, random forest feature importances, and `best_model_pipeline.joblib` containing the fitted preprocessing and estimator together. The best model is selected by lowest held-out RMSE. Metrics and interpretations should be read from the generated artifacts; no results are assumed in advance.

## Plots

Price distribution; price by manufacturing year and distance; median price by fuel, transmission, and brand; numeric correlation heatmap; IQR boxplots; and held-out MAE/RMSE comparison. Category plots use medians to reduce the influence of skewed prices. The brand chart includes brands with at least ten observations and shows the fifteen highest medians for readability.
