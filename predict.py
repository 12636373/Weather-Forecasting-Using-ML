"""
Interactive Weather Predictor CLI
Author: Harish (@12636373)
GitHub: https://github.com/12636373/Weather-Forecasting-Using-ML
"""

import argparse
import sys
import os

# Ensure safe UTF-8 terminal encoding on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

WEATHER_ICONS = {
    "drizzle": "[Drizzle]",
    "fog": "[Fog]",
    "rain": "[Rain]",
    "snow": "[Snow]",
    "sun": "[Sun]",
}

def load_and_train():
    data_path = 'seattle-weather.csv'
    if not os.path.exists(data_path):
        data_path = os.path.join('..', 'input', 'weather-prediction', 'seattle-weather.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError("Dataset 'seattle-weather.csv' not found. Please place it in the project root.")

    df = pd.read_csv(data_path)
    le = LabelEncoder()
    df['weather_encoded'] = le.fit_transform(df['weather'])
    
    features = ["precipitation", "temp_max", "temp_min", "wind"]
    X = df[features]
    y = df['weather_encoded']

    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)

    # Train models
    dt = DecisionTreeRegressor(random_state=1).fit(train_X, train_y)
    rf = RandomForestRegressor(random_state=1).fit(train_X, train_y)
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.04, random_state=1).fit(train_X, train_y)

    return le, dt, rf, xgb

def predict_weather(precip, tmax, tmin, wind, le, dt, rf, xgb):
    input_data = pd.DataFrame([{
        "precipitation": float(precip),
        "temp_max": float(tmax),
        "temp_min": float(tmin),
        "wind": float(wind)
    }])

    pred_dt = dt.predict(input_data)[0]
    pred_rf = rf.predict(input_data)[0]
    pred_xgb = xgb.predict(input_data)[0]

    def decode(val):
        idx = int(round(np.clip(val, 0, len(le.classes_) - 1)))
        label = le.classes_[idx]
        return label, WEATHER_ICONS.get(label, label)

    dt_label, dt_display = decode(pred_dt)
    rf_label, rf_display = decode(pred_rf)
    xgb_label, xgb_display = decode(pred_xgb)

    print("\n" + "="*50)
    print("WEATHER FORECAST PREDICTION RESULTS")
    print("="*50)
    print("Inputs:")
    print(f"  * Precipitation:     {precip:.1f} mm")
    print(f"  * Max Temperature:   {tmax:.1f} °C")
    print(f"  * Min Temperature:   {tmin:.1f} °C")
    print(f"  * Wind Speed:        {wind:.1f} m/s")
    print("-" * 50)
    print("Predictions:")
    print(f"  * Decision Tree:     {dt_display} (Score: {pred_dt:.2f})")
    print(f"  * Random Forest:     {rf_display} (Score: {pred_rf:.2f})")
    print(f"  * XGBoost (Best):    {xgb_display} (Score: {pred_xgb:.2f})")
    print("="*50)
    print(f"Consensus Forecast: {xgb_display}")
    print("="*50 + "\n")

def interactive_mode(le, dt, rf, xgb):
    print("\n" + "="*50)
    print("Welcome to the Interactive Weather Predictor!")
    print("Author: Harish (@12636373)")
    print("="*50)
    try:
        precip = float(input("Enter expected precipitation (mm, e.g. 2.5): ").strip())
        tmax = float(input("Enter maximum temperature (°C, e.g. 18.0): ").strip())
        tmin = float(input("Enter minimum temperature (°C, e.g. 9.5): ").strip())
        wind = float(input("Enter average wind speed (m/s, e.g. 3.2): ").strip())
        predict_weather(precip, tmax, tmin, wind, le, dt, rf, xgb)
    except ValueError:
        print("Error: Please enter valid numerical values.")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting interactive predictor.")

def main():
    parser = argparse.ArgumentParser(description="Predict weather from meteorological features using ML models.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive prompt mode")
    parser.add_argument("--precipitation", "-p", type=float, help="Precipitation in mm")
    parser.add_argument("--temp_max", "-tmax", type=float, help="Maximum temperature in °C")
    parser.add_argument("--temp_min", "-tmin", type=float, help="Minimum temperature in °C")
    parser.add_argument("--wind", "-w", type=float, help="Wind speed in m/s")

    args = parser.parse_args()

    le, dt, rf, xgb = load_and_train()

    if args.interactive or (args.precipitation is None and args.temp_max is None):
        interactive_mode(le, dt, rf, xgb)
    else:
        if None in (args.precipitation, args.temp_max, args.temp_min, args.wind):
            print("Error: Please supply all 4 arguments (--precipitation, --temp_max, --temp_min, --wind) or use --interactive.")
            sys.exit(1)
        predict_weather(args.precipitation, args.temp_max, args.temp_min, args.wind, le, dt, rf, xgb)

if __name__ == "__main__":
    main()
