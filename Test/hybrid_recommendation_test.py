import joblib
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from Model import HybridRecommendation
from TF_Object import L2NormalizeLayer
from TF_Object import CollaborativeFilteringLayer
from TF_Object import CollaborativeFilteringModel


class HybridRecommendationTest(unittest.TestCase):
    def setUp(self) -> None:
        # Important variables based on dataset
        self.num_users = 300
        self.num_tourism = 437
        self.num_features = 64

        # Load Data
        self.distance_df = pd.read_csv('../Dataset/tourist_spots_distance.csv')
        self.user_data_vecs = np.load('../Vector/user_vector.npy')
        self.tourism_data_vecs = np.load('../Vector/tourism_vector.npy')
        self.Y_mean = np.load('../Vector/Y_mean.npy')

        # Load Scaler
        self.user_scaler = joblib.load('../Scaler/user_scaler.gz')
        self.tourism_scaler = joblib.load('../Scaler/tourism_scaler.gz')
        self.target_scaler = joblib.load('../Scaler/target_scaler.gz')

        # Load Collaborative Filtering Model
        self.cofi_model = tf.keras.models.load_model('../Model/collaborative_filtering.h5', custom_objects={
            'CollaborativeFilteringModel': CollaborativeFilteringModel,
            'CollaborativeFilteringLayer': CollaborativeFilteringLayer
        })

        # Load Content-Based Model
        self.cb_model = tf.keras.models.load_model('../Model/content_based.h5', custom_objects={
            'L2NormalizeLayer': L2NormalizeLayer
        })

        self.hybrid_model = HybridRecommendation(
            cofi_model=self.cofi_model,
            cb_model=self.cb_model,
            user_scaler=self.user_scaler,
            tourism_scaler=self.tourism_scaler,
            target_scaler=self.target_scaler,
            Y_mean=self.Y_mean
        )

    def test_model_architecture(self) -> None:
        """Testing the collaborative & content-based model architecture"""

        # Collaborative Filtering
        # Assert model
        self.assertIsInstance(self.cofi_model, tf.keras.Model)
        self.assertIsInstance(self.cofi_model, CollaborativeFilteringModel)

        # Assert layer
        self.assertEqual(len(self.cofi_model.layers), 1)
        self.assertIsInstance(self.cofi_model.layers[0], CollaborativeFilteringLayer)

        # Assert weights (X, W, b)
        self.assertEqual(len(self.cofi_model.layers[0].get_weights()), 3)
        self.assertEqual(self.cofi_model.layers[0].get_weights()[0].shape, (self.num_tourism, self.num_features))
        self.assertEqual(self.cofi_model.layers[0].get_weights()[1].shape, (self.num_users + 1, self.num_features))
        self.assertEqual(self.cofi_model.layers[0].get_weights()[2].shape, (1, self.num_users + 1))
        # self.num_users + 1, because a `general user` (ID = 0) has been created in the model creation

        # Content Based (CB)
        cb_main_architecture = {
            0: tf.keras.layers.InputLayer,
            1: tf.keras.layers.InputLayer,
            2: tf.keras.Sequential,
            3: tf.keras.Sequential,
            4: L2NormalizeLayer,
            5: L2NormalizeLayer,
            6: tf.keras.layers.Dot
        }

        cb_sequential_architecture = {
            0: tf.keras.layers.Dense,
            1: tf.keras.layers.Dropout,
            2: tf.keras.layers.Dense,
            3: tf.keras.layers.Dense,
        }

        self.assertIsInstance(self.cb_model, tf.keras.Model)
        self.assertEqual(len(self.cb_model.layers), 7)
        for i in range(len(self.cb_model.layers)):
            self.assertIsInstance(self.cb_model.layers[i], cb_main_architecture[i])
            if self.cb_model.layers[i] is cb_main_architecture[i]:
                for j in range(len(self.cb_model.layers[i].layers)):
                    self.assertIsInstance(self.cb_model.layers[i].layers[j], cb_sequential_architecture[j])

    def test_normal_hybrid_recommendation(self) -> None:
        """Testing the model's ability to make predictions"""
        # Prepare user data
        user_id = 120
        current_user_data = self.user_data_vecs[user_id]

        # Get (normal) hybrid recommendation for that user
        recommended_id = self.hybrid_model.get_recommendation(
            user_id,
            current_user_data,
            self.tourism_data_vecs
        )

        # Assert prediction results
        self.assertEqual(len(recommended_id), self.num_tourism)

    def test_distance_hybrid_recommendation(self) -> None:
        """Testing the model's ability to make predictions"""
        # Prepare data
        user_id = 120
        chosen_spot_id = 300
        current_user_data = self.user_data_vecs[user_id]
        chosen_spot_distance_df = self.distance_df.\
            loc[self.distance_df['Place_Id_Source'] == chosen_spot_id]

        # Get (distance-based) hybrid recommendation for that user
        recommended_id = self.hybrid_model.get_recommendation(
            user_id,
            current_user_data,
            self.tourism_data_vecs,
            chosen_spot_id,
            chosen_spot_distance_df
        )

        # Assert prediction results
        # self.num_tourism - 1, because chosen_spot_id = 300 does not include
        self.assertEqual(len(recommended_id), self.num_tourism - 1)


if __name__ == '__main__':
    unittest.main()
