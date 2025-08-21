import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

INPUT_DIM = 190+64+1 # 190 for state, 64 for action, 1 for action type

class ResidualNetwork(tf.keras.Model):
    def __init__(self, input_dim=INPUT_DIM):
        super(ResidualNetwork, self).__init__()

    def predict_rewards(self, game_state, action):
        encoded = tf.expand_dims(game_state.encode(action), axis=0)
        reward = self(encoded, training=False) 
        print(f"Predicted reward: {reward}")
        return float(reward)