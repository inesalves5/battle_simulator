import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

INPUT_DIM = 190+64+1 # 190 for state, 64 for action, 1 for action type

class FullyConnectedNetwork(tf.keras.Model):
    def __init__(self, input_dim=INPUT_DIM):
        super(FullyConnectedNetwork, self).__init__()
        self.model = models.Sequential([
            layers.Input(shape=(input_dim,)),     # entrada
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='linear')  # saída com 1 valor: resultado do Japanese
        ])
    
    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def predict_rewards(self, game_state, action):
        encoded = tf.convert_to_tensor(game_state.encode(action), dtype=tf.float16)
        encoded = tf.expand_dims(encoded, axis=0)
        reward = self(encoded, training=False)
        return float(reward.numpy()[0][0])