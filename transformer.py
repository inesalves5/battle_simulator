<<<<<<< HEAD
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
=======
import tensorflow as tf
from tensorflow.keras import layers, Model
from keras_nlp.layers import TransformerEncoder

INPUT_DIM = 190+64+1 # 190 for state, 64 for action, 1 for action type

class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        # Multi-head self attention
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        
        # Feedforward
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        
        # Normalização
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Atenção com residual
        attn_output = self.att(inputs, inputs)  
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  

        # Feedforward com residual
        ffn_output = self.ffn(out1)  
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerNetwork(tf.keras.Model):
    def __init__(self, input_dim=INPUT_DIM, embed_dim=256, num_heads=4, ff_dim=512, num_blocks=2):
        super(TransformerNetwork, self).__init__()
        
        # Projeta entrada densa para embedding
        self.embedding = layers.Dense(embed_dim)

        # Vários blocos encoder
        self.encoders = [TransformerEncoderBlock(embed_dim, num_heads, ff_dim) 
                         for _ in range(num_blocks)]
        
        # Pooling para reduzir sequência → vetor fixo
        self.global_pool = layers.GlobalAveragePooling1D()

        # Saída final
        self.output_layer = layers.Dense(1, activation="linear")

    def call(self, inputs, training=False):
        # (batch, features) → (batch, seq_len=1, embed_dim)
        x = self.embedding(inputs)
        x = tf.expand_dims(x, axis=1)

        # Passa pelos encoders
        for encoder in self.encoders:
            x = encoder(x, training=training)

        # Pooling para vetor
        x = self.global_pool(x)
        return self.output_layer(x)
>>>>>>> e11e087 (graficos e mlp e resnet feitos)
