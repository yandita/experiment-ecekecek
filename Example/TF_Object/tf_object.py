import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="ContentBased")
class L2NormalizeLayer(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super(L2NormalizeLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis)

    def get_config(self):
        return {'axis': self.axis}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='CollaborativeFiltering')
class CollaborativeFilteringLayer(tf.keras.layers.Layer):
    def __init__(self, num_users, num_tourism, num_features, name='collaborative_filtering_layer', **kwargs):
        super(CollaborativeFilteringLayer, self).__init__(name=name, **kwargs)
        self.num_users = num_users
        self.num_tourism = num_tourism
        self.num_features = num_features

        self.X = self.add_weight(
            shape=(self.num_tourism, self.num_features),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            dtype=tf.float32,
            trainable=True,
            name='X'
        )

        self.W = self.add_weight(
            shape=(self.num_users, self.num_features),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            dtype=tf.float32,
            trainable=True,
            name='W'
        )

        self.b = self.add_weight(
            shape=(1, self.num_users),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            dtype=tf.float32,
            trainable=True,
            name='b'
        )

    def call(self, inputs):  # inputs = user_id
        if tf.math.equal(inputs, tf.constant(-1, dtype=tf.int32)):
            return tf.matmul(self.X, self.W, transpose_b=True) + self.b
        else:
            return tf.matmul(self.X, tf.reshape(self.W[inputs], (-1, 1))) + self.b[:, inputs]

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_users': self.num_users,
            'num_tourism': self.num_tourism,
            'num_features': self.num_features
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='CollaborativeFiltering')
class CollaborativeFilteringModel(tf.keras.Model):
    def __init__(self, num_users, num_tourism, num_features, name='collaborative_filtering_model', **kwargs):
        super(CollaborativeFilteringModel, self).__init__(name=name, **kwargs)
        self.num_users = num_users
        self.num_tourism = num_tourism
        self.num_features = num_features
        self.collaborative_filtering = CollaborativeFilteringLayer(num_users, num_tourism, num_features)

    def call(self, inputs):
        return self.collaborative_filtering(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_users': self.num_users,
            'num_tourism': self.num_tourism,
            'num_features': self.num_features
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
