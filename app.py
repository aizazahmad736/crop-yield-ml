from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load Trained Model and Scaler
model = joblib.load('model/crop_model.pkl')
scaler = joblib.load('model/scaler.pkl')
feature_importances = joblib.load('model/feature_importances.pkl')

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = [
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]
        
        # Preprocess features
        features_scaled = scaler.transform([features])
        
        # Predict class probabilities
        probs = model.predict_proba(features_scaled)[0]
        classes = model.classes_
        
        # Rank recommendations
        crop_probs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        
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
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
