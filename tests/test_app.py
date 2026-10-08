import unittest
from unittest.mock import patch

import app as app_module


class CropAdvisorApiTests(unittest.TestCase):
    def setUp(self):
        app_module.app.config['TESTING'] = True
        self.client = app_module.app.test_client()
        self.valid_payload = {
            'N': 90,
            'P': 45,
            'K': 40,
            'temperature': 25,
            'humidity': 85,
            'ph': 6.5,
            'rainfall': 240,
        }

    def test_homepage_loads(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Crop Suitability Advisor', response.data)

    def test_health_reports_ready_when_assets_are_loaded(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['status'], 'ok')

    def test_model_info_lists_supported_classes(self):
        response = self.client.get('/model-info')
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload['success'])
        self.assertEqual(payload['supported_crops'], len(app_module.model.classes_))
        self.assertEqual(payload['features'], app_module.FEATURE_COLUMNS)
        self.assertIsNotNone(payload['evaluation'])
        self.assertGreaterEqual(payload['evaluation']['accuracy'], 0)
        self.assertLessEqual(payload['evaluation']['accuracy'], 1)

    def test_predict_returns_ranked_recommendations(self):
        response = self.client.post('/predict', json=self.valid_payload)
        payload = response.get_json()
        recommendations = payload['top_recommendations']

        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload['success'])
        self.assertEqual(len(recommendations), 4)
        self.assertEqual(payload['recommended_crop'], recommendations[0]['name'])
        scores = [item['confidence'] for item in recommendations]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_predict_rejects_missing_fields(self):
        response = self.client.post('/predict', json={'N': 90})
        self.assertEqual(response.status_code, 400)
        self.assertIn('Missing required field', response.get_json()['error'])

    def test_predict_rejects_values_outside_training_range(self):
        for field, (_, maximum) in app_module.FEATURE_BOUNDS.items():
            with self.subTest(field=field):
                payload = dict(self.valid_payload, **{field: maximum + 1})
                response = self.client.post('/predict', json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn(field, response.get_json()['error'])

    def test_predict_rejects_non_finite_values(self):
        payload = dict(self.valid_payload, humidity='NaN')
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn('finite', response.get_json()['error'])

    def test_health_reports_unavailable_model(self):
        with patch.object(app_module, 'model', None), patch.object(app_module, 'scaler', None):
            response = self.client.get('/health')
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()['status'], 'missing_model')


if __name__ == '__main__':
    unittest.main()
