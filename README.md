# AgriMind ML - Crop Yield & Suitability Predictor 🌿

A complete Machine Learning application that recommends optimal crop types based on soil parameters (N, P, K, pH) and environmental conditions (Temperature, Humidity, Rainfall).

## 🚀 Features
- **Machine Learning Engine**: Random Forest Classifier trained on 2,200 agricultural samples across 22 crop categories (99.7% test accuracy).
- **Feature Importance Analysis**: Analyzes which climate/soil factors impact crop success most.
- **REST API**: Flask backend providing real-time crop suitability predictions and confidence scores.
- **Interactive UI**: Glassmorphism web interface with parameter sliders, soil presets, and real-time Chart.js visualizers.

<img width="953" height="446" alt="Screenshot 2026-07-21 155238" src="https://github.com/user-attachments/assets/c16ab639-47ec-4a40-86f2-4074f0d539b8" />

## 🛠️ Project Structure
```text
crop-yield-ml/
├── data/
│   ├── generate_dataset.py     # Synthesizes agricultural dataset
│   └── crop_recommendation.csv  # 2,200 crop records
├── model/
│   ├── crop_model.pkl          # Trained Random Forest model
│   ├── scaler.pkl              # Standard Scaler instance
│   └── feature_importances.pkl # Calculated feature weights
├── static/
│   ├── css/style.css           # Glassmorphism styling
│   └── js/main.js              # Fetch & Chart.js logic
├── templates/
│   └── index.html              # Dashboard interface
├── app.py                      # Flask REST API & Web Server
└── train_model.py              # ML Model training script
```

## 🧪 How to Run
1. Run the Flask Server:
```bash
python app.py
```
2. Open your browser and navigate to: `http://127.0.0.1:5000`
