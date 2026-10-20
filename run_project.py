"""
Weather Forecasting Using Machine Learning
Pipeline runner and model benchmark script.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Dataset
data_path = 'seattle-weather.csv'
if not os.path.exists(data_path):
    data_path = os.path.join('..', 'input', 'weather-prediction', 'seattle-weather.csv')

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

print("\n" + "="*50)
print("1. DATASET OVERVIEW")
print("="*50)
print(f"Total Records: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values Count:")
print(df.isnull().sum())

print("\nWeather Condition Frequency:")
print(df['weather'].value_counts())

# 2. Pre-processing & Label Encoding
print("\n" + "="*50)
print("2. PRE-PROCESSING & LABEL ENCODING")
print("="*50)
le = LabelEncoder()
df['weather_encoded'] = le.fit_transform(df['weather'])

label_mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
inv_mapping = {idx: cls for idx, cls in enumerate(le.classes_)}
print("Label Encoding mapping:")
for cls, idx in label_mapping.items():
    print(f"  {cls} -> {idx}")

# Generate visualizations
os.makedirs('outputs', exist_ok=True)
os.makedirs('assets', exist_ok=True)

def save_fig(name):
    p1 = os.path.join('outputs', name)
    p2 = os.path.join('assets', name)
    plt.savefig(p1, dpi=300)
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved figure to '{p1}' and '{p2}'")

# Heatmap
plt.figure(figsize=(8, 6))
corr_matrix = df.drop(columns=['date', 'weather']).corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
save_fig('correlation_heatmap.png')

# Weather class distribution
plt.figure(figsize=(7, 4))
palette = sns.color_palette("viridis", len(df['weather'].unique()))
sns.countplot(data=df, x='weather', hue='weather', order=df['weather'].value_counts().index, palette=palette, legend=False)
plt.title("Distribution of Weather Categories in Seattle", fontsize=13, fontweight="bold")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Days")
plt.tight_layout()
save_fig('weather_distribution.png')

# 3. Model Training & Evaluation
print("\n" + "="*50)
print("3. MODEL TRAINING & COMPARISON")
print("="*50)

features = ["precipitation", "temp_max", "temp_min", "wind"]
X = df[features]
y = df['weather_encoded']

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)
print(f"Training set: {train_X.shape[0]} samples")
print(f"Testing set:  {test_X.shape[0]} samples")

results = []

# Model 1: Decision Tree Regressor
model1 = DecisionTreeRegressor(random_state=1)
model1.fit(train_X, train_y)
pred1 = model1.predict(test_X)
mae1 = mean_absolute_error(test_y, pred1)
rmse1 = np.sqrt(mean_squared_error(test_y, pred1))
r2_1 = r2_score(test_y, pred1)
results.append({"Model": "Decision Tree Regressor", "MAE": mae1, "RMSE": rmse1, "R2": r2_1, "ModelObj": model1, "Predictions": pred1})

# Model 2: Random Forest Regressor
model2 = RandomForestRegressor(random_state=1)
model2.fit(train_X, train_y)
pred2 = model2.predict(test_X)
mae2 = mean_absolute_error(test_y, pred2)
rmse2 = np.sqrt(mean_squared_error(test_y, pred2))
r2_2 = r2_score(test_y, pred2)
results.append({"Model": "Random Forest Regressor", "MAE": mae2, "RMSE": rmse2, "R2": r2_2, "ModelObj": model2, "Predictions": pred2})

# Model 3: XGBoost Regressor
# Notebook version: model3 = XGBRegressor(n_estimators=100, learning_rate=0.04)
model3 = XGBRegressor(n_estimators=100, learning_rate=0.04, random_state=1)
model3.fit(train_X, train_y)
pred3 = model3.predict(test_X)
mae3 = mean_absolute_error(test_y, pred3)
rmse3 = np.sqrt(mean_squared_error(test_y, pred3))
r2_3 = r2_score(test_y, pred3)
results.append({"Model": "Extreme Gradient Boosting (XGB)", "MAE": mae3, "RMSE": rmse3, "R2": r2_3, "ModelObj": model3, "Predictions": pred3})

# Results table
results_df = pd.DataFrame(results)[["Model", "MAE", "RMSE", "R2"]]
print("\nBenchmark Summary:")
print(results_df.to_string(index=False))

# Best model
best_model_info = min(results, key=lambda x: x["MAE"])
print(f"\nBest Model: {best_model_info['Model']} with MAE = {best_model_info['MAE']:.6f}")

# Model comparison plot
plt.figure(figsize=(8, 5))
bars = plt.bar(results_df["Model"], results_df["MAE"], color=['#4C72B0', '#55A868', '#C44E52'], width=0.5)
plt.ylabel("Mean Absolute Error (Lower is Better)", fontsize=11)
plt.title("Model Comparison: Weather Forecasting Precision (MAE)", fontsize=13, fontweight="bold")
plt.ylim(0, max(results_df["MAE"]) * 1.25)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
save_fig('model_comparison.png')

# Feature Importances plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
feat_imp_rf = pd.Series(model2.feature_importances_, index=features).sort_values(ascending=True)
feat_imp_rf.plot(kind='barh', color='#55A868')
plt.title("Random Forest Feature Importance", fontsize=11, fontweight="bold")
plt.xlabel("Importance")

plt.subplot(1, 2, 2)
feat_imp_xgb = pd.Series(model3.feature_importances_, index=features).sort_values(ascending=True)
feat_imp_xgb.plot(kind='barh', color='#C44E52')
plt.title("XGBoost Feature Importance", fontsize=11, fontweight="bold")
plt.xlabel("Importance")
plt.tight_layout()
save_fig('feature_importance.png')

# Sample predictions comparison
print("\n" + "="*50)
print("4. SAMPLE PREDICTIONS ON TEST DATA")
print("="*50)
sample_test = test_X.head(10).copy()
sample_test['Actual Weather'] = [inv_mapping[val] for val in test_y.head(10)]
sample_test['DT Pred'] = [f"{p:.2f} ({inv_mapping[int(round(np.clip(p, 0, 4)))]})" for p in pred1[:10]]
sample_test['RF Pred'] = [f"{p:.2f} ({inv_mapping[int(round(np.clip(p, 0, 4)))]})" for p in pred2[:10]]
sample_test['XGB Pred'] = [f"{p:.2f} ({inv_mapping[int(round(np.clip(p, 0, 4)))]})" for p in pred3[:10]]
print(sample_test[['precipitation', 'temp_max', 'temp_min', 'wind', 'Actual Weather', 'XGB Pred']])

print("\n" + "="*50)
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
print("="*50)
