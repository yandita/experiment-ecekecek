import sklearn
import numpy as np
import pandas as pd
import tensorflow as tf


class HybridRecommendation:
    def __init__(
            self,
            cofi_model: tf.keras.Model,
            cb_model: tf.keras.Model,
            user_scaler: sklearn.base.BaseEstimator,
            tourism_scaler: sklearn.base.BaseEstimator,
            target_scaler: sklearn.base.BaseEstimator,
            Y_mean: np.ndarray
    ):
        self.cofi_model = cofi_model
        self.cb_model = cb_model
        self.user_scaler = user_scaler
        self.tourism_scaler = tourism_scaler
        self.target_scaler = target_scaler
        self.Y_mean = Y_mean
    
    def __hybrid(
            self,
            user_id: int,
            current_user_data: np.ndarray,
            tourism_data_vecs: np.ndarray
    ) -> dict:
        # COLLABORATIVE FILTERING
        # Prepare user data
        user_id = tf.constant(user_id, dtype=tf.int32)
        # Make prediction
        cofi_y_pred = self.cofi_model(user_id)
        # Convert to Numpy and restore the mean
        cofi_y_pred = cofi_y_pred.numpy() + self.Y_mean
        # Get the index of the predicted result sequence
        sorted_cofi_index = np.argsort(cofi_y_pred, axis=0).reshape(-1)
        # Create cofi_point, where the highest recommended tourist destination ID will get the highest points. 
        cofi_points = dict()
        for point, idx in enumerate(sorted_cofi_index):
            cofi_points[int(idx)] = {
                'point': point,
                'rating': cofi_y_pred[int(idx)].item()
            }

        # CONTENT-BASED
        # Prepare user data
        current_user_vecs = np.tile(current_user_data, (tourism_data_vecs.shape[0], 1))
        scaled_current_user_vecs = self.user_scaler.transform(current_user_vecs)
        # Prepare Tourism Data
        scaled_tourism_vecs = self.tourism_scaler.transform(tourism_data_vecs)
        # Make prediction
        cb_y_pred_norm = self.cb_model.predict([scaled_current_user_vecs, scaled_tourism_vecs])
        cb_y_pred = self.target_scaler.inverse_transform(cb_y_pred_norm)
        # Get the index of the predicted result sequence
        sorted_cb_index = np.argsort(cb_y_pred_norm, axis=0).reshape(-1)
        # Create cb_points, where the highest recommended tourist destination ID will get the highest points. 
        cb_points = dict()        
        for point, idx in enumerate(sorted_cb_index):
            cb_points[int(idx)] = {
                'point': point,
                'rating': cb_y_pred[int(idx)].item()
            }

        # HYBRID WEIGHT
        # Calculate the total point value based on collaborative and content-based weighting
        final_points = dict()
        for tourism_idx in range(tourism_data_vecs.shape[0]):
            final_points[tourism_idx] = {
                'point': cofi_points[tourism_idx]['point'] * 0.5 + cb_points[tourism_idx]['point'] * 0.5,
                'cofi_rating': cofi_points[tourism_idx]['rating'],
                'cb_rating': cb_points[tourism_idx]['rating'],
            }

        # Return tourism ID from most recommended to least recommended
        return {key + 1: value for key, value in final_points.items()}
    
    def get_recommendation(
            self,
            user_id: int,
            current_user_data: np.ndarray,
            tourism_data_vecs: np.ndarray,
            choosen_spot_id: int | None = None,
            distance_from_chosen_spot_df: pd.DataFrame | None = None
    ) -> dict | list:
        
        # Get hybrid recommendation
        recommended_spots = self.__hybrid(user_id, current_user_data, tourism_data_vecs)

        # Get recommendation based on spot distance
        if choosen_spot_id is not None and distance_from_chosen_spot_df is not None:
            sorted_distance_from_chosen_spot_df = distance_from_chosen_spot_df.\
                sort_values(by='Distance', ascending=False)
            
            distance_points = dict()
            for point, idx in enumerate(sorted_distance_from_chosen_spot_df['Place_Id_Target']):
                distance_points[idx] = {
                    'point': point,
                    'distance': sorted_distance_from_chosen_spot_df.\
                        loc[sorted_distance_from_chosen_spot_df['Place_Id_Target'] == idx, 'Distance'].values
                }
        
            final_points_with_distance = dict()
            for tourism_idx in [i for i in range(1, tourism_data_vecs.shape[0] + 1) if i != choosen_spot_id]:
                final_points_with_distance[tourism_idx] = {
                    'point': recommended_spots[tourism_idx]['point'] * 0.55 + distance_points[tourism_idx]['point'] * 0.45,
                    'cofi_rating': recommended_spots[tourism_idx]['cofi_rating'],
                    'cb_rating': recommended_spots[tourism_idx]['cb_rating'],
                    'distance': distance_points[tourism_idx]['distance'].item()
                }
            
            # Sorts the dictionary by point value from largest to smallest
            sorted_final_points_with_distance = dict(sorted(final_points_with_distance.items(),
                                                            key=lambda x: x[1]['point'], reverse=True))
                
            # Return dictionary, with key = Place_Id and value = Distance, from most recommended
            return {key: value['distance'] for key, value in sorted_final_points_with_distance.items()}
        
        else:
            # Sorts the dictionary by point value from largest to smallest
            sorted_final_points = dict(sorted(recommended_spots.items(), key=lambda x: x[1]['point'], reverse=True))

            # Returns a list of Place_Ids from most recommended
            return list(sorted_final_points.keys())
