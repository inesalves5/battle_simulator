import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

TOTAL_UNITS = 190         # dimensão do estado
ACTION_DIM = 2           # número de ações possíveis por jogador (ajuste conforme necessário)

class ValueMLP(tf.keras.Model):
    def __init__(self, state_dim=TOTAL_UNITS, action_dim=ACTION_DIM):
        super(ValueMLP, self).__init__()
        input_dim = state_dim + action_dim

        self.model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(2)  # saída: [r1, r2]
        ])

    def call(self, s, a1, a2):
        # s: [batch_size, state_dim]
        # a1, a2: [batch_size, action_dim] (one-hot ou vetores contínuos)
        x = tf.concat([s, a1, a2], axis=-1)
        return self.model(x)

    def predict_rewards(self, game_state, a1_vec, a2_vec):
        # game_state.encode() deve retornar vetor shape [state_dim]
        s = tf.expand_dims(game_state.encode(), axis=0)
        a1 = tf.expand_dims(a1_vec, axis=0)
        a2 = tf.expand_dims(a2_vec, axis=0)

        rewards = self(s, a1, a2, training=False)  # shape [1, 2]
        r1, r2 = rewards[0][0].numpy(), rewards[0][1].numpy()
        print(f"Predicted rewards: r1 = {r1}, r2 = {r2}")
        return float(r1), float(r2)

"""input_dim = TOTAL_UNITS + 2 * ACTION_DIM
inputs = tf.keras.Input(shape=(input_dim,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(2)(x)
keras_model = tf.keras.Model(inputs, outputs)
keras_model.save('model.h5')"""

# Converter para TFLite
def convert_model_to_tflite():
    model = tf.keras.models.load_model('model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('converted_model.tflite', 'wb') as f:
        f.write(tflite_model)