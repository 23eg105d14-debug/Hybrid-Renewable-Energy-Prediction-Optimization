# -------------------------------
# FILE: data_generator.py
# -------------------------------
# Generates a synthetic hybrid (solar + wind) dataset and saves it to data/hybrid_energy_dataset.csv

import os
import numpy as np
import pandas as pd

np.random.seed(0)

def generate_dataset(n_hours=1000, start_date='2025-01-01'):
    dates = pd.date_range(start_date, periods=n_hours, freq='H')

    temperature = np.random.uniform(0, 45, n_hours)
    humidity = np.random.uniform(10, 100, n_hours)
    solar_irradiance = np.random.uniform(0, 1100, n_hours)
    wind_speed = np.random.uniform(0, 20, n_hours)
    cloud_cover = np.random.uniform(0, 1, n_hours)
    pressure = np.random.uniform(980, 1035, n_hours)

    # Simple physics-inspired solar model: efficiency depends on temp and cloud cover
    solar_efficiency = 0.18 * (1 - 0.005 * (temperature - 25)) * (1 - 0.4 * cloud_cover)
    solar_output = solar_irradiance * solar_efficiency * np.random.uniform(0.9, 1.05, n_hours) / 1000

    # Wind model using cubic dependence with air density correction
    air_density = 1.225 * (1 - 0.0036 * (temperature - 15))
    wind_output = 0.5 * air_density * (wind_speed ** 3) * 0.0001 * np.random.uniform(0.85, 1.1, n_hours)

    total_output = solar_output + wind_output
    total_output = np.maximum(total_output, 0)

    df = pd.DataFrame({
        'Datetime': dates,
        'Temperature_C': np.round(temperature, 2),
        'Humidity_%': np.round(humidity, 2),
        'Solar_Irradiance_W_m2': np.round(solar_irradiance, 2),
        'Wind_Speed_m_s': np.round(wind_speed, 2),
        'Cloud_Cover': np.round(cloud_cover, 3),
        'Pressure_hPa': np.round(pressure, 2),
        'Solar_Output_kW': np.round(solar_output, 4),
        'Wind_Output_kW': np.round(wind_output, 4),
        'Total_Output_kW': np.round(total_output, 4)
    })

    os.makedirs('data', exist_ok=True)
    out_path = os.path.join('data', 'hybrid_energy_dataset.csv')
    df.to_csv(out_path, index=False)
    return out_path, df

if __name__ == '__main__':
    path, _ = generate_dataset()
    print('Dataset generated:', path)


# -------------------------------
# FILE: train_model.py
# -------------------------------
# Loads dataset, trains two regression models (RandomForest & GradientBoosting),
# evaluates them, saves models and results to results/ directory.

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(path='data/hybrid_energy_dataset.csv'):
    df = pd.read_csv(path, parse_dates=['Datetime'])
    return df

def train_and_evaluate(df, features=None, target='Total_Output_kW', test_size=0.2, random_state=42):
    if features is None:
        features = ['Temperature_C','Humidity_%','Solar_Irradiance_W_m2','Wind_Speed_m_s','Cloud_Cover','Pressure_hPa']

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    rf = RandomForestRegressor(n_estimators=150, random_state=random_state)
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=random_state)

    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    y_pred_gb = gb.predict(X_test)

    def metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2

    rf_mae, rf_rmse, rf_r2 = metrics(y_test, y_pred_rf)
    gb_mae, gb_rmse, gb_r2 = metrics(y_test, y_pred_gb)

    perf = pd.DataFrame({
        'Model': ['RandomForest', 'GradientBoosting'],
        'MAE_kW': [round(rf_mae, 5), round(gb_mae, 5)],
        'RMSE_kW': [round(rf_rmse, 5), round(gb_rmse, 5)],
        'R2': [round(rf_r2, 5), round(gb_r2, 5)]
    })

    os.makedirs('results', exist_ok=True)
    perf.to_csv(os.path.join('results', 'model_performance.csv'), index=False)

    comparison = pd.DataFrame({
        'Datetime': df.loc[X_test.index, 'Datetime'].values,
        'Actual_kW': y_test.values,
        'Pred_RF_kW': y_pred_rf,
        'Pred_GB_kW': y_pred_gb
    }).reset_index(drop=True)
    comparison.to_csv(os.path.join('results', 'results_comparison.csv'), index=False)

    joblib.dump(rf, os.path.join('results', 'rf_model.joblib'))
    joblib.dump(gb, os.path.join('results', 'gb_model.joblib'))

    return perf, comparison

if __name__ == '__main__':
    df = load_data()
    perf, comparison = train_and_evaluate(df)
    print('Training complete. Performance:')
    print(perf)
    print('Results saved to results/')


# -------------------------------
# FILE: predict.py
# -------------------------------
# Simple prediction utility that loads a saved model and predicts on either
# - a single sample provided via command-line arguments, or
# - a CSV file with the same feature columns.

import argparse
import pandas as pd
import joblib
import os

FEATURES = ['Temperature_C','Humidity_%','Solar_Irradiance_W_m2','Wind_Speed_m_s','Cloud_Cover','Pressure_hPa']

def load_model(model_name='rf'):
    model_path = os.path.join('results', f"{model_name}_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')
    return joblib.load(model_path)

def predict_from_csv(model, csv_path, out_path=None):
    df = pd.read_csv(csv_path)
    X = df[FEATURES]
    preds = model.predict(X)
    df['Predicted_Total_kW'] = preds
    if out_path:
        df.to_csv(out_path, index=False)
    return df

def predict_single(model, values_dict):
    row = pd.DataFrame([values_dict])[FEATURES]
    pred = model.predict(row)[0]
    return float(pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict total hybrid energy output using saved model')
    parser.add_argument('--model', choices=['rf','gb'], default='rf', help='model to use: rf or gb')
    parser.add_argument('--csv', type=str, help='CSV file with input features (will output CSV with predictions)')
    parser.add_argument('--out', type=str, help='Optional output CSV path (when using --csv)')
    parser.add_argument('--single', action='store_true', help='Provide single sample via CLI args')
    parser.add_argument('--Temperature_C', type=float)
    parser.add_argument('--Humidity_%', type=float)
    parser.add_argument('--Solar_Irradiance_W_m2', type=float)
    parser.add_argument('--Wind_Speed_m_s', type=float)
    parser.add_argument('--Cloud_Cover', type=float)
    parser.add_argument('--Pressure_hPa', type=float)

    args = parser.parse_args()

    mdl = load_model(args.model)

    if args.csv:
        out = predict_from_csv(mdl, args.csv, out_path=args.out)
        print('Predictions saved' if args.out else 'Predictions computed (not saved)')
        print(out.head())
    elif args.single:
        req = {f: getattr(args, f) for f in FEATURES}
        if any(v is None for v in req.values()):
            raise ValueError('All feature values must be provided for single prediction')
        pred = predict_single(mdl, req)
        print('Predicted Total_Output_kW:', round(pred, 6))
    else:
        raise ValueError('Either --csv or --single with all features must be provided')


# -------------------------------
# FILE: requirements.txt
# -------------------------------
# Minimal dependencies required to run the scripts and notebook

pandas
numpy
scikit-learn
matplotlib
joblib

# End of hybrid script bundle
