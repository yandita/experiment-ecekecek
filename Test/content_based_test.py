import joblib
import unittest
import numpy as np
import tensorflow as tf
from TF_Object import L2NormalizeLayer


class ContentBasedTest(unittest.TestCase):
    def setUp(self) -> None:
        # Load Vector Data
        self.user_data_vecs = np.load('../Vector/user_vector.npy')
        self.tourism_data_vecs = np.load('../Vector/tourism_vector.npy')

        # Load Scaler
        self.user_scaler = joblib.load('../Scaler/user_scaler.gz')
        self.tourism_scaler = joblib.load('../Scaler/tourism_scaler.gz')
        self.target_scaler = joblib.load('../Scaler/target_scaler.gz')

        # Load NN Model
        self.model = tf.keras.models.load_model('../Model/content_based.h5', custom_objects={
            'L2NormalizeLayer': L2NormalizeLayer
        })

    def test_model_architecture(self) -> None:
        """Testing the content-based model architecture"""
        main_architecture = {
            0: tf.keras.layers.InputLayer,
            1: tf.keras.layers.InputLayer,
            2: tf.keras.Sequential,
            3: tf.keras.Sequential,
            4: L2NormalizeLayer,
            5: L2NormalizeLayer,
            6: tf.keras.layers.Dot
        }

        sequential_architecture = {
            0: tf.keras.layers.Dense,
            1: tf.keras.layers.Dropout,
            2: tf.keras.layers.Dense,
            3: tf.keras.layers.Dense,
        }

        self.assertIsInstance(self.model, tf.keras.Model)
        self.assertEqual(len(self.model.layers), 7)
        for i in range(len(self.model.layers)):
            self.assertIsInstance(self.model.layers[i], main_architecture[i])
            if self.model.layers[i] is main_architecture[i]:
                for j in range(len(self.model.layers[i].layers)):
                    self.assertIsInstance(self.model.layers[i].layers[j], sequential_architecture[j])

    def test_recommendation(self) -> None:
        """Testing the model's ability to make predictions"""
        # Prepare User Data
        user_id = 120
        current_user_data = self.user_data_vecs[user_id]
        current_user_vecs = np.tile(current_user_data, (self.tourism_data_vecs.shape[0], 1))
        scaled_current_user_vecs = self.user_scaler.transform(current_user_vecs)

        # Prepare Tourism Data
        scaled_tourism_vecs = self.tourism_scaler.transform(self.tourism_data_vecs)

        # Make prediction
        y_pred = self.model.predict([scaled_current_user_vecs, scaled_tourism_vecs])
        y_pred = self.target_scaler.inverse_transform(y_pred)

        # Sort the prediction results
        sorted_index = np.argsort(-y_pred, axis=0).reshape(-1)
        sorted_y_pred = y_pred[sorted_index].reshape(-1)

        # Assert prediction results
        self.assertEqual(len(sorted_y_pred), len(self.tourism_data_vecs))
        self.assertTrue(np.all(sorted_y_pred >= 0))
        self.assertTrue(np.all(sorted_y_pred <= 5))


if __name__ == '__main__':
    unittest.main()
