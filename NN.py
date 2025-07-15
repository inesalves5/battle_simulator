import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

TOTAL_UNITS = 190 # Total number of units in the real game

class ValueMLP(tf.keras.Model):
    def __init__(self, input_dim=TOTAL_UNITS):
        super(ValueMLP, self).__init__()
        self.model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

    def call(self, x):
        x = self.model(x)
        return tf.squeeze(x, axis=-1)

    def predict_value(self, game_state):
        x = tf.expand_dims(game_state.encode(), axis=0)  # add batch dimension
        value = self(x, training=False)  # call forward pass
        print("Predicted value:", value)
        return float(value.numpy()[0])

def convert_model_to_tflite():
    model = tf.keras.models.load_model('model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('converted_model.tflite', 'wb') as f:
        f.write(tflite_model)
