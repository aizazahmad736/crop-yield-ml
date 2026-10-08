from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'model'
FEATURE_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

CROP_DETAILS = {
    'rice': {'category': 'Cereal', 'season': 'Kharif', 'water': 'High', 'icon': '🌾'},
    'maize': {'category': 'Cereal', 'season': 'Kharif/Rabi', 'water': 'Medium', 'icon': '🌽'},
    'chickpea': {'category': 'Pulses', 'season': 'Rabi', 'water': 'Low', 'icon': '🫘'},
    'kidneybeans': {'category': 'Pulses', 'season': 'Kharif', 'water': 'Medium', 'icon': '🫘'},
    'pigeonpeas': {'category': 'Pulses', 'season': 'Kharif', 'water': 'Low-Medium', 'icon': '🌱'},
    'mothbeans': {'category': 'Pulses', 'season': 'Kharif', 'water': 'Low', 'icon': '🌱'},
    'mungbean': {'category': 'Pulses', 'season': 'Kharif/Summer', 'water': 'Low', 'icon': '🌱'},
    'blackgram': {'category': 'Pulses', 'season': 'Kharif', 'water': 'Low-Medium', 'icon': '🌱'},
    'lentil': {'category': 'Pulses', 'season': 'Rabi', 'water': 'Low', 'icon': '🌱'},
    'pomegranate': {'category': 'Fruit', 'season': 'Perennial', 'water': 'Medium', 'icon': '🍎'},
    'banana': {'category': 'Fruit', 'season': 'Perennial', 'water': 'High', 'icon': '🍌'},
    'mango': {'category': 'Fruit', 'season': 'Summer', 'water': 'Medium', 'icon': '🥭'},
    'grapes': {'category': 'Fruit', 'season': 'Rabi/Summer', 'water': 'Medium', 'icon': '🍇'},
    'watermelon': {'category': 'Fruit', 'season': 'Summer', 'water': 'Medium', 'icon': '🍉'},
    'muskmelon': {'category': 'Fruit', 'season': 'Summer', 'water': 'Low-Medium', 'icon': '🍈'},
    'apple': {'category': 'Fruit', 'season': 'Temperate/Autumn', 'water': 'Medium-High', 'icon': '🍎'},
    'orange': {'category': 'Fruit', 'season': 'Winter/Perennial', 'water': 'Medium', 'icon': '🍊'},
    'papaya': {'category': 'Fruit', 'season': 'Perennial', 'water': 'Medium', 'icon': '🍈'},
    'coconut': {'category': 'Cash Crop', 'season': 'Perennial', 'water': 'High', 'icon': '🥥'},
    'cotton': {'category': 'Cash Crop', 'season': 'Kharif', 'water': 'Medium', 'icon': '☁️'},
    'jute': {'category': 'Cash Crop', 'season': 'Kharif', 'water': 'High', 'icon': '🌾'},
    'coffee': {'category': 'Cash Crop', 'season': 'Perennial', 'water': 'High', 'icon': '☕'}
}


def load_model_assets():
    model_path = MODEL_DIR / 'crop_model.pkl'
    scaler_path = MODEL_DIR / 'scaler.pkl'
    feature_importances_path = MODEL_DIR / 'feature_importances.pkl'

    missing_files = [
        path.name for path in (model_path, scaler_path, feature_importances_path)
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Model files are missing: " + ', '.join(missing_files) + ". Run 'python train_model.py' first."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_importances = joblib.load(feature_importances_path)
    return model, scaler, feature_importances


try:
    model, scaler, feature_importances = load_model_assets()
    app_load_error = None
except FileNotFoundError as exc:
    model = None
    scaler = None
    feature_importances = {}
    app_load_error = str(exc)

app = Flask(__name__)


def parse_input_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError('Request body must be a JSON object.')

    missing_fields = [field for field in FEATURE_COLUMNS if field not in payload]
    if missing_fields:
        raise ValueError(f"Missing required field(s): {', '.join(missing_fields)}")

    values = {}
    for field in FEATURE_COLUMNS:
        raw_value = payload.get(field)
        if raw_value is None or str(raw_value).strip() == '':
            raise ValueError(f"Field '{field}' is required.")

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"Field '{field}' must be numeric.")

        if not np.isfinite(value):
            raise ValueError(f"Field '{field}' must be finite.")

        values[field] = value

    if values['ph'] < 0 or values['ph'] > 14:
        raise ValueError('Soil pH must be between 0 and 14.')
    if values['humidity'] < 0 or values['humidity'] > 100:
        raise ValueError('Humidity percentage must be between 0 and 100.')
    if values['temperature'] < -10 or values['temperature'] > 60:
        raise ValueError('Temperature must be between -10°C and 60°C.')

    return values


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'ok' if model is not None and scaler is not None else 'missing_model',
        'message': app_load_error or 'Model ready.'
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'success': False, 'error': app_load_error or 'Model is not available.'}), 503

    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict()

    try:
        values = parse_input_payload(payload)
        features = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        features_scaled = scaler.transform(features)

        probs = model.predict_proba(features_scaled)[0]
        classes = model.classes_
        crop_probs = sorted(zip(classes, probs), key=lambda item: item[1], reverse=True)
        top_crop, top_confidence = crop_probs[0]

        top_3 = []
        for crop_name, confidence in crop_probs[:4]:
            meta = CROP_DETAILS.get(crop_name, {'category': 'General', 'season': 'All', 'water': 'Medium', 'icon': '🌱'})
            top_3.append({
                'name': crop_name.capitalize(),
                'confidence': round(confidence * 100, 1),
                'category': meta['category'],
                'season': meta['season'],
                'water': meta['water'],
                'icon': meta['icon']
            })

        return jsonify({
            'success': True,
            'recommended_crop': top_crop.capitalize(),
            'confidence': round(top_confidence * 100, 1),
            'top_recommendations': top_3,
            'feature_importance': feature_importances
        })
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'success': False, 'error': f'Prediction failed: {exc}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
