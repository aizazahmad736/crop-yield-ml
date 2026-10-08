import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data' / 'crop_recommendation.csv'
MODEL_DIR = BASE_DIR / 'model'
FEATURE_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET_COLUMN = 'label'


def main():
    print('--- Step 1: Loading Dataset ---')
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f'Dataset not found at {DATA_PATH}. Run data/generate_dataset.py first.'
        )

    df = pd.read_csv(DATA_PATH)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f'Dataset is missing required columns: {", ".join(missing_columns)}')
    if df.empty or df[required_columns].isnull().any().any():
        raise ValueError('Dataset must contain rows without missing feature or label values.')

    print(f'Dataset shape: {df.shape}')
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print('\n--- Step 2: Feature Scaling ---')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print('\n--- Step 3: Training Random Forest Classifier ---')
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
    model.fit(X_train_scaled, y_train)

    print('\n--- Step 4: Evaluating on Held-out Test Data ---')
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Macro F1: {macro_f1 * 100:.2f}%')
    print(f'Weighted F1: {weighted_f1 * 100:.2f}%')

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    metrics = {
        'accuracy': round(float(accuracy), 4),
        'macro_f1': round(float(macro_f1), 4),
        'weighted_f1': round(float(weighted_f1), 4),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'crop_count': int(y.nunique()),
        'evaluated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'per_crop': {
            crop: {
                'precision': round(float(scores['precision']), 4),
                'recall': round(float(scores['recall']), 4),
                'f1_score': round(float(scores['f1-score']), 4),
                'support': int(scores['support'])
            }
            for crop, scores in report.items()
            if crop in model.classes_
        }
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / 'crop_model.pkl')
    joblib.dump(scaler, MODEL_DIR / 'scaler.pkl')

    importances = {
        feature: round(float(value), 4)
        for feature, value in zip(FEATURE_COLUMNS, model.feature_importances_)
    }
    joblib.dump(importances, MODEL_DIR / 'feature_importances.pkl')
    with (MODEL_DIR / 'model_metrics.json').open('w', encoding='utf-8') as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f'\nFeature Importances: {importances}')
    print(f"Saved model, scaler, feature importances, and evaluation metrics in '{MODEL_DIR}'.")


if __name__ == '__main__':
    main()
