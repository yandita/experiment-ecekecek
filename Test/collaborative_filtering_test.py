import unittest
import numpy as np
import tensorflow as tf
from TF_Object import CollaborativeFilteringLayer
from TF_Object import CollaborativeFilteringModel


class CollaborativeFilteringTest(unittest.TestCase):
    def setUp(self) -> None:
        # Important variables based on dataset
        self.num_users = 300
        self.num_tourism = 437
        self.num_features = 64

        # Y_mean for denormalize prediction results
        self.Y_mean = np.load('../Vector/Y_mean.npy')

        # Load NN Model
        self.model = tf.keras.models.load_model('../Model/collaborative_filtering.h5', custom_objects={
            'CollaborativeFilteringModel': CollaborativeFilteringModel,
            'CollaborativeFilteringLayer': CollaborativeFilteringLayer
        })

    def test_model_architecture(self) -> None:
        """Testing the collaborative filtering model architecture"""
        # Assert model
        self.assertIsInstance(self.model, tf.keras.Model)
        self.assertIsInstance(self.model, CollaborativeFilteringModel)

        # Assert layer
        self.assertEqual(len(self.model.layers), 1)
        self.assertIsInstance(self.model.layers[0], CollaborativeFilteringLayer)

        # Assert weights (X, W, b)
        self.assertEqual(len(self.model.layers[0].get_weights()), 3)
        self.assertEqual(self.model.layers[0].get_weights()[0].shape, (self.num_tourism, self.num_features))
        self.assertEqual(self.model.layers[0].get_weights()[1].shape, (self.num_users + 1, self.num_features))
        self.assertEqual(self.model.layers[0].get_weights()[2].shape, (1, self.num_users + 1))
        # self.num_users + 1, because a `general user` (ID = 0) has been created in the model creation

    def test_recommendation(self) -> None:
        """Testing the model's ability to make predictions"""
        # Prepare User Data
        user_id = 120
        user_id = tf.constant(user_id, dtype=tf.int32)

        # Make prediction
        my_pred = self.model(user_id)

        # Convert to Numpy and restore the mean
        my_pred = my_pred.numpy()
        my_pred = my_pred + self.Y_mean

        # Sort the prediction results
        sorted_index = np.argsort(-my_pred, axis=0).reshape(-1)
        sorted_my_pred = my_pred[sorted_index].reshape(-1)

        # Assert prediction results
        self.assertEqual(len(sorted_my_pred), self.num_tourism)
        self.assertTrue(np.all(sorted_my_pred >= 0))
        self.assertTrue(np.all(sorted_my_pred <= 5))


if __name__ == '__main__':
    unittest.main()
