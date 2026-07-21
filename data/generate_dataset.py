import numpy as np
import pandas as pd
import os

# Set seed for reproducibility
np.random.seed(42)

crops = {
    'rice': {'N': (60, 100), 'P': (35, 60), 'K': (35, 45), 'temp': (20, 27), 'humidity': (80, 90), 'ph': (6.0, 7.0), 'rainfall': (180, 300)},
    'maize': {'N': (60, 100), 'P': (35, 60), 'K': (15, 25), 'temp': (18, 27), 'humidity': (55, 75), 'ph': (5.8, 7.3), 'rainfall': (60, 110)},
    'chickpea': {'N': (20, 50), 'P': (55, 80), 'K': (75, 85), 'temp': (17, 20), 'humidity': (14, 20), 'ph': (6.0, 8.5), 'rainfall': (65, 95)},
    'kidneybeans': {'N': (15, 40), 'P': (55, 80), 'K': (15, 25), 'temp': (15, 24), 'humidity': (18, 25), 'ph': (5.5, 6.0), 'rainfall': (60, 150)},
    'pigeonpeas': {'N': (15, 40), 'P': (55, 75), 'K': (15, 25), 'temp': (27, 38), 'humidity': (40, 65), 'ph': (5.0, 7.5), 'rainfall': (90, 180)},
    'mothbeans': {'N': (15, 40), 'P': (35, 60), 'K': (15, 25), 'temp': (24, 32), 'humidity': (40, 65), 'ph': (3.5, 10.0), 'rainfall': (30, 75)},
    'mungbean': {'N': (15, 40), 'P': (35, 60), 'K': (15, 25), 'temp': (27, 30), 'humidity': (80, 90), 'ph': (6.2, 7.2), 'rainfall': (35, 60)},
    'blackgram': {'N': (35, 60), 'P': (55, 80), 'K': (15, 25), 'temp': (25, 35), 'humidity': (60, 75), 'ph': (6.5, 7.8), 'rainfall': (60, 75)},
    'lentil': {'N': (15, 40), 'P': (55, 80), 'K': (15, 25), 'temp': (18, 30), 'humidity': (60, 70), 'ph': (5.9, 7.0), 'rainfall': (35, 55)},
    'pomegranate': {'N': (15, 40), 'P': (10, 30), 'K': (35, 45), 'temp': (18, 25), 'humidity': (85, 95), 'ph': (5.5, 7.2), 'rainfall': (100, 112)},
    'banana': {'N': (80, 120), 'P': (70, 95), 'K': (45, 55), 'temp': (25, 30), 'humidity': (75, 85), 'ph': (5.5, 6.5), 'rainfall': (90, 120)},
    'mango': {'N': (15, 40), 'P': (15, 40), 'K': (25, 35), 'temp': (27, 36), 'humidity': (45, 55), 'ph': (4.5, 7.0), 'rainfall': (80, 100)},
    'grapes': {'N': (100, 140), 'P': (120, 145), 'K': (195, 205), 'temp': (8, 40), 'humidity': (80, 90), 'ph': (5.5, 6.5), 'rainfall': (60, 75)},
    'watermelon': {'N': (80, 120), 'P': (5, 30), 'K': (45, 55), 'temp': (24, 27), 'humidity': (80, 90), 'ph': (6.0, 7.0), 'rainfall': (40, 60)},
    'muskmelon': {'N': (80, 120), 'P': (5, 30), 'K': (45, 55), 'temp': (27, 29), 'humidity': (90, 95), 'ph': (6.0, 6.8), 'rainfall': (20, 30)},
    'apple': {'N': (0, 40), 'P': (120, 145), 'K': (195, 205), 'temp': (21, 24), 'humidity': (90, 95), 'ph': (5.5, 6.5), 'rainfall': (100, 125)},
    'orange': {'N': (0, 40), 'P': (5, 30), 'K': (5, 15), 'temp': (10, 35), 'humidity': (90, 95), 'ph': (6.0, 7.5), 'rainfall': (100, 120)},
    'papaya': {'N': (35, 70), 'P': (45, 70), 'K': (45, 55), 'temp': (23, 44), 'humidity': (90, 95), 'ph': (6.5, 7.0), 'rainfall': (40, 250)},
    'coconut': {'N': (15, 40), 'P': (5, 30), 'K': (25, 35), 'temp': (25, 28), 'humidity': (90, 99), 'ph': (5.5, 6.5), 'rainfall': (130, 225)},
    'cotton': {'N': (100, 140), 'P': (35, 60), 'K': (15, 25), 'temp': (22, 26), 'humidity': (75, 85), 'ph': (5.8, 8.0), 'rainfall': (60, 90)},
    'jute': {'N': (60, 100), 'P': (35, 60), 'K': (35, 45), 'temp': (23, 26), 'humidity': (70, 85), 'ph': (6.0, 7.4), 'rainfall': (150, 200)},
    'coffee': {'N': (80, 120), 'P': (15, 35), 'K': (25, 35), 'temp': (23, 28), 'humidity': (50, 70), 'ph': (6.0, 7.2), 'rainfall': (115, 200)}
}

records_per_crop = 100
data = []

for crop, params in crops.items():
    for _ in range(records_per_crop):
        n = float(np.clip(np.random.normal(np.mean(params['N']), (params['N'][1] - params['N'][0]) / 4), 0, 140))
        p = float(np.clip(np.random.normal(np.mean(params['P']), (params['P'][1] - params['P'][0]) / 4), 5, 145))
        k = float(np.clip(np.random.normal(np.mean(params['K']), (params['K'][1] - params['K'][0]) / 4), 5, 205))
        temp = float(np.clip(np.random.normal(np.mean(params['temp']), (params['temp'][1] - params['temp'][0]) / 4), 8, 45))
        humidity = float(np.clip(np.random.normal(np.mean(params['humidity']), (params['humidity'][1] - params['humidity'][0]) / 4), 14, 100))
        ph = float(np.clip(np.random.normal(np.mean(params['ph']), (params['ph'][1] - params['ph'][0]) / 4), 3.5, 10.0))
        rainfall = float(np.clip(np.random.normal(np.mean(params['rainfall']), (params['rainfall'][1] - params['rainfall'][0]) / 4), 20, 300))
        
        data.append({
            'N': round(n, 2),
            'P': round(p, 2),
            'K': round(k, 2),
            'temperature': round(temp, 2),
            'humidity': round(humidity, 2),
            'ph': round(ph, 2),
            'rainfall': round(rainfall, 2),
            'label': crop
        })

df = pd.DataFrame(data)
os.makedirs('data', exist_ok=True)
df.to_csv('data/crop_recommendation.csv', index=False)
print(f"Successfully generated {len(df)} samples across {len(crops)} crop categories into data/crop_recommendation.csv")
