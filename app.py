import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'model'
FEATURE_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
FEATURE_BOUNDS = {
    'N': (0, 140),
    'P': (5, 145),
    'K': (5, 205),
    'temperature': (8, 45),
    'humidity': (14, 100),
    'ph': (3.5, 10),
    'rainfall': (20, 300),
}

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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024


def load_model_assets():
    model_path = MODEL_DIR / 'crop_model.pkl'
    scaler_path = MODEL_DIR / 'scaler.pkl'
    feature_importances_path = MODEL_DIR / 'feature_importances.pkl'
    metrics_path = MODEL_DIR / 'model_metrics.json'

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
    metrics = None
    if metrics_path.exists():
        with metrics_path.open(encoding='utf-8') as metrics_file:
            metrics = json.load(metrics_file)

    return model, scaler, feature_importances, metrics


model = None
scaler = None
feature_importances = {}
model_metrics = None
app_load_error = None

try:
    model, scaler, feature_importances, model_metrics = load_model_assets()
except Exception as exc:
    app_load_error = str(exc)
    app.logger.exception('Unable to load crop prediction model assets.')


def parse_input_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError('Request body must be a JSON object.')

    missing_fields = [field for field in FEATURE_COLUMNS if field not in payload]
    if missing_fields:
        raise ValueError(f"Missing required field(s): {', '.join(missing_fields)}")

    values = {}
    for field in FEATURE_COLUMNS:
        raw_value = payload.get(field)
        if isinstance(raw_value, bool) or raw_value is None or str(raw_value).strip() == '':
            raise ValueError(f"Field '{field}' must be a number.")

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"Field '{field}' must be numeric.")

        if not np.isfinite(value):
            raise ValueError(f"Field '{field}' must be finite.")

        minimum, maximum = FEATURE_BOUNDS[field]
        if value < minimum or value > maximum:
            raise ValueError(f"Field '{field}' must be between {minimum} and {maximum}.")

        values[field] = value

    return values


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health')
def health_check():
    is_ready = model is not None and scaler is not None
    return jsonify({
        'status': 'ok' if is_ready else 'missing_model',
        'message': app_load_error or 'Model ready.'
    }), (200 if is_ready else 503)


@app.route('/model-info')
def model_info():
    if model is None or scaler is None:
        return jsonify({'success': False, 'error': app_load_error or 'Model is not available.'}), 503

    return jsonify({
        'success': True,
        'algorithm': type(model).__name__,
        'supported_crops': len(model.classes_),
        'features': FEATURE_COLUMNS,
        'feature_importance': feature_importances,
        'evaluation': model_metrics
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'success': False, 'error': app_load_error or 'Model is not available.'}), 503

    payload = request.get_json(silent=True) if request.is_json else request.form.to_dict()
    if payload is None:
        return jsonify({'success': False, 'error': 'Request body must contain valid JSON.'}), 400

    try:
        values = parse_input_payload(payload)
        features = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        features_scaled = scaler.transform(features)

        probs = model.predict_proba(features_scaled)[0]
        crop_probs = sorted(zip(model.classes_, probs), key=lambda item: item[1], reverse=True)
        top_crop, top_confidence = crop_probs[0]

        recommendations = []
        for crop_name, confidence in crop_probs[:4]:
            meta = CROP_DETAILS.get(crop_name, {
                'category': 'General',
                'season': 'All',
                'water': 'Medium',
                'icon': '🌱'
            })
            recommendations.append({
                'name': crop_name.capitalize(),
                'confidence': round(float(confidence) * 100, 1),
                'category': meta['category'],
                'season': meta['season'],
                'water': meta['water'],
                'icon': meta['icon']
            })

        return jsonify({
            'success': True,
            'recommended_crop': top_crop.capitalize(),
            'confidence': round(float(top_confidence) * 100, 1),
            'top_recommendations': recommendations,
            'feature_importance': feature_importances
        })
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception:
        app.logger.exception('Crop prediction failed.')
        return jsonify({'success': False, 'error': 'Prediction failed. Check the server log for details.'}), 500


if __name__ == '__main__':
    app.run(
        debug=os.environ.get('FLASK_DEBUG') == '1',
        host=os.environ.get('FLASK_HOST', '127.0.0.1'),
        port=int(os.environ.get('PORT', '5000'))
    )
