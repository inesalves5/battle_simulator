import tensorflow as tf
from tensorflow.keras import layers, Model
from keras_nlp.layers import TransformerEncoder

class GameTransformer(Model):
    def __init__(self, state_dim=1, action_vocab=200, action_embed_dim=4, max_units=120, transformer_dim=64, num_heads=4, ff_dim=128):
        super().__init__()

        self.action_embed = layers.Embedding(input_dim=action_vocab, output_dim=action_embed_dim, mask_zero=False)

        self.transformer_layer = tf.keras.Sequential([
            TransformerEncoder(
                intermediate_dim=ff_dim,
                num_heads=num_heads,
            ),
            TransformerEncoder(
                intermediate_dim=ff_dim,
                num_heads=num_heads,
            )
        ])

        self.output_head = tf.keras.Sequential([
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'), #devia ser linar pq valores neg e pos
            layers.Dense(2)  # [r1, r2]
        ])

    def call(self, x):
        states = x[:, :, 0:1]               # damage
        actions = tf.cast(x[:, :, 1], tf.int32)  # target

        embedded_actions = self.action_embed(actions)
        x = tf.concat([states, embedded_actions], axis=-1)

        x = self.transformer_layer(x)
        return self.output_head(x)

import json
import numpy as np
import tensorflow as tf
import csv

def load_dataset():
    data = []
    with open("res.json", "r") as f:
        for line in f:
            d = json.loads(line)
            enc = d["game"]
            r = d["result"]
            data.append((enc, r))

    features = np.stack([x[0] for x in data])
    labels = np.stack([x[1] for x in data])
    return tf.data.Dataset.from_tensor_slices((features, labels))

def load_partial_dataset(l):
    data = []
    with open("res.json", "r") as f:
        for line in f:
            if l > 0:
                l -= 1
            else:
                break
            d = json.loads(line)
            enc = d["game"]
            r = d["result"]
            data.append((enc, r))

    features = np.stack([x[0] for x in data])
    labels = np.stack([x[1] for x in data])
    return tf.data.Dataset.from_tensor_slices((features, labels))

def train_model(epochs=20, batch_size=32, lr=1e-3):
    dataset = load_partial_dataset(epochs * batch_size * 10)  # Load enough data for training
    dataset = dataset.shuffle(10000).batch(batch_size)

    with open("nn_train_log.csv", "a", newline="") as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(["epoch", "loss", "predicted_result"])

        for epoch in range(epochs):
            game_case = main.get_test_case()
            enc = game_case.encode()
            
            model = GameTransformer()
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

            try:
                model.load_weights("game_transformer.weights.h5")
                print("Pesos carregados.")
            except:
                print("Iniciando com pesos aleatórios.")

            predicted_result = predict_result(model, enc)
            history = model.fit(dataset, epochs=1, verbose=0)
            loss = history.history["loss"][0]
            print(f"Época {epoch+1}/{epochs} - Loss: {loss:.4f}, Predicted: {predicted_result}")
            log_writer.writerow([epoch + 1, loss, predicted_result])

        model.save_weights("game_transformer.weights.h5")
    return model

def predict_result(model, x):
    x = tf.expand_dims(x, axis=0)   # adiciona batch
    result = model(x, training=False)
    return result.numpy()[0]        # retorna [r1, r2]
import main

if __name__ == "__main__":
    model = train_model(epochs=50, batch_size=32, lr=1e-3)
