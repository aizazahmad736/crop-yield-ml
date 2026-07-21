import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("--- Step 1: Loading Dataset ---")
df = pd.read_csv('data/crop_recommendation.csv')
print(f"Dataset shape: {df.shape}")

# Features & Target
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- Step 2: Feature Scaling ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Step 3: Training Random Forest Classifier ---")
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
model.fit(X_train_scaled, y_train)

print("\n--- Step 4: Model Evaluation ---")
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%")

# Save model and scaler
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/crop_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# Feature Importances
importances = dict(zip(X.columns, [round(val, 4) for val in model.feature_importances_]))
joblib.dump(importances, 'model/feature_importances.pkl')

print(f"\nFeature Importances: {importances}")
print("\n--- Step 5: Model Saved to model/crop_model.pkl & model/scaler.pkl ---")
