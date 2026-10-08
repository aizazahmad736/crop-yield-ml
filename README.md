# AgriMind ML - Crop Suitability Advisor 🌿

A machine-learning application that ranks crop options using soil nutrients (N, P, K, pH) and environmental conditions (temperature, humidity, rainfall). It is a suitability demo, not a crop-yield forecast or a substitute for local agronomic advice.

## 🚀 Features
- **Machine-learning engine**: Random Forest classifier supports 22 crop classes using seven soil and climate features.
- **Evaluation metrics**: Training reports held-out accuracy, macro/weighted F1, and per-crop precision, recall, and F1. The dashboard shows the latest evaluation summary.
- **Input validation**: API values are checked for numeric validity and the feature ranges covered by the generated training dataset.
- **Exportable history**: Recent predictions are stored in the current browser and can be exported as CSV, including inputs and ranked alternatives.
- **Model health and metadata**: `/health` reports readiness and `/model-info` provides model details, evaluation metrics, and feature importances.
- **Responsive interface**: Soil/climate presets, ranked alternatives, feature-importance visualization, loading and error feedback, and keyboard focus styling.

> **Important:** The included dataset is synthetically generated. High test scores on synthetic data do not establish real-world accuracy, crop yield, or suitability for a particular farm. Prediction scores are not calibrated probabilities.

<img width="953" height="446" alt="Screenshot 2026-07-21 155238" src="https://github.com/user-attachments/assets/c16ab639-47ec-4a40-86f2-4074f0d539b8" />

<img width="923" height="402" alt="Screenshot 2026-07-21 155252" src="https://github.com/user-attachments/assets/911eeb15-8cf3-4182-b751-09bcf5e7bc31" />

<img width="931" height="404" alt="Screenshot 2026-07-21 155312" src="https://github.com/user-attachments/assets/ea2d3cc6-e52a-4e5b-930a-ee2bdbf8f3bd" />

## 🛠️ Project Structure
```text
crop-yield-ml/
├── data/
│   ├── generate_dataset.py     # Synthesizes agricultural dataset
│   └── crop_recommendation.csv  # 2,200 crop records
├── model/
│   ├── crop_model.pkl          # Trained Random Forest model
│   ├── scaler.pkl              # Standard Scaler instance
│   ├── feature_importances.pkl # Calculated feature weights
│   └── model_metrics.json      # Held-out evaluation metrics
├── tests/
│   └── test_app.py             # Flask API and input-validation tests
├── static/
│   ├── css/style.css           # Glassmorphism styling
│   └── js/main.js              # Fetch & Chart.js logic
├── templates/
│   └── index.html              # Dashboard interface
├── app.py                      # Flask REST API & Web Server
└── train_model.py              # ML Model training script
```

## 🧪 How to Run
1. Create and activate a virtual environment, then install dependencies:
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
2. Generate the model and held-out evaluation report:
```bash
python train_model.py
```
3. Start the local development server:
```bash
python app.py
```
4. Open your browser and navigate to: `http://127.0.0.1:5000`
5. Run the test suite:
```bash
python -m unittest discover -s tests -v
```

The server binds to `127.0.0.1` by default and debug mode is off. Set `FLASK_DEBUG=1` only for local development. Configure `FLASK_HOST` and `PORT` as needed; use a production WSGI server when deploying.

## API

- `POST /predict` — accepts a JSON object with `N`, `P`, `K`, `temperature`, `humidity`, `ph`, and `rainfall`; returns the top four crop classes and their model scores.
- `GET /health` — returns model readiness (`200` when ready, `503` when model assets are unavailable).
- `GET /model-info` — returns algorithm, supported crop count, feature importance, and latest held-out evaluation metrics.
