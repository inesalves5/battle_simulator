import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

INPUT_DIM = 190+64+1 # 190 for state, 64 for action, 1 for action type

class ResidualBlock(layers.Layer):
    def __init__(self, units):
        super(ResidualBlock, self).__init__()
        self.dense1 = layers.Dense(units, activation='relu')
        self.dense2 = layers.Dense(units)
        self.activation = layers.Activation('relu')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.activation(x + inputs)

class ResidualNetwork(tf.keras.Model):
    def __init__(self, input_dim=INPUT_DIM):
        super(ResidualNetwork, self).__init__()
        self.fc_input = layers.Dense(256, activation='relu')
        
        #blocos residuais
        self.residual_block1 = ResidualBlock(256)
        self.residual_block2 = ResidualBlock(256)
        
        self.fc_middle = layers.Dense(128, activation='relu')
        self.output_layer = layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.fc_input(inputs)

        #blocos residuais
        x = self.residual_block1(x)
        x = self.residual_block2(x)

        x = self.fc_middle(x)
        return self.output_layer(x)

    def predict_rewards(self, game_state, action):
        encoded = tf.expand_dims(game_state.encode(action), axis=0)
        reward = self(encoded, training=False) 
        return float(reward.numpy()[0][0])
