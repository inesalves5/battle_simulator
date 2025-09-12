import tensorflow as tf
import numpy as np
import copy
import random
from collections import Counter
import main 
from itertools import product
import json
import fully_connected
import resnet

options = list(product(range(1, 11), repeat=2))
INPUT_DIM = 190+64+1 # 190 for state, 64 for action, 1 for action type
nn_type = "resnet"  # or "fully_connected"

def serialize_example(feature, target):
    # Ensure feature is numpy
    feature = np.array(feature, dtype=np.float16)  
    feature_bytes = feature.tobytes()

    target_val = float(target)  # ensure scalar
    
    feature_dict = {
        "features": tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature_bytes])),
        "targets": tf.train.Feature(float_list=tf.train.FloatList(value=[target_val])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example_proto.SerializeToString()


def write_tfrecord(data, filename="dataset.tfrecord"):
    with tf.io.TFRecordWriter(filename) as writer:
        for game_state, reward in data:
            if tf.is_tensor(game_state):
                game_state = game_state.numpy()
            example = serialize_example(game_state, reward)
            writer.write(example)

def parse_example(example_proto, num_features=255):
    feature_description = {
        "features": tf.io.FixedLenFeature([], tf.string),   # stored as raw bytes
        "targets": tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    # Decode features back to float16 → cast to float32 for training
    features = tf.io.decode_raw(parsed["features"], tf.float16)
    features = tf.reshape(features, (num_features,))
    features = tf.cast(features, tf.float32)

    target = parsed["targets"]
    return features, target

def load_dataset(filename="dataset.tfrecord", num_features=255, batch_size=64):
    raw_dataset = tf.data.TFRecordDataset(filename)
    dataset = raw_dataset.map(lambda x: parse_example(x, num_features))
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

class GameValueDataset(tf.data.Dataset):
    def __new__(cls, data):
        #write_tfrecord(data)
        return load_dataset()

class PredictAfterEpoch(tf.keras.callbacks.Callback):
    def __init__(self, test_inputs="test_cases_encoded.json", log_path=f"{nn_type}/test_cases.json"): #logpath depends on nn type
        super().__init__()
        self.test_inputs = [l.strip() for l in open(test_inputs)]  # shape: (n_samples, n_features)
        self.log_path = log_path

    def on_epoch_end(self, epoch, logs=None):
        for i in range(len(self.test_inputs)):
            if isinstance(self.test_inputs[i], str):
                fixed = json.loads(self.test_inputs[i])   # converte de string para lista
            else:
                fixed = self.test_inputs[i]
            x = np.array(fixed, dtype=np.float32)  # garante float
            x = np.expand_dims(x, axis=0)  # adiciona batch dimension
            prediction = self.model.predict(x, verbose=0).flatten()
            with open(self.log_path, "a", encoding="utf-8") as f:
                json.dump({"epoch": epoch + 1, "sample": i, "prediction": float(prediction)}, f)
                f.write("\n")

def train_value_net(epochs=40, batch_size=64, lr=1e-4):
    data = [] #para quando nao esta a ser escrito
    dataset = GameValueDataset(data)
    dataset = dataset.map(lambda x, y: (tf.reshape(x, [tf.shape(x)[0], 255]), y))
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))
    model = resnet.ResidualNetwork(INPUT_DIM) # depends on nn type
    dummy_input = tf.zeros((1, INPUT_DIM))
    _ = model(dummy_input)  # constrói o modelo    
    try:
        model.load_weights(f"{nn_type}.weights.h5") # depends on nn type
    except Exception as e:
        print("erro:", e)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse')
    print("compiled")
    callbacks = [PredictAfterEpoch()]
    history = model.fit(dataset, epochs=epochs, callbacks=callbacks)
    print("correu")
    with open(f"{nn_type}/training.json", "a", encoding="utf-8") as file:  # depends on nn type
        json.dump(history.history, file)
    model.save_weights(f'{nn_type}.weights.h5') #depends on nn type
    return model

def simulate(game_state, iterations, action):
    r = 0
    buffer = []
    g = copy.deepcopy(game_state)
    max_reward = g.max_reward(action)
    _, _, actions, _ = main.mcts_round(g, max_reward=max_reward,iterations=iterations*100)
    for action in actions:
        g, reward, _ = g.get_next_state(action)
        if g is None:
            return None, None
        r += reward[0]
    buffer.append((None, g))
    r += g.reward_zone()[0]
    return r, buffer

def self_play_and_generate_training_data():
    with open("replay_buffer.json", "a", encoding="utf-8") as f:
        for n in options:
            a, j = n
            print("-----------------------case", n)
            game_state_day, game_state_night = main.create_random_game_pair(a, j)
            r_day, buffer_day = simulate(game_state_day, iterations=a+j, action="day")
            r_night, buffer_night = simulate(game_state_night, iterations=a+j, action="night")
            if r_day is None or r_night is None:
                continue
            all_games_day, all_games_night = [], []
            for (a0, game_part) in buffer_day:
                all_games_day += game_part.generate_equivalent_games(a0)
            for (a0, g) in all_games_day:
                features = g.encode(a0).numpy().tolist()
                f.write(json.dumps({"game": features, "result": r_day}) + "\n")
            for (a0, game_part) in buffer_night:
                all_games_night += game_part.generate_equivalent_games(a0)
            for (a0, g) in all_games_night:
                features = g.encode(a0).numpy().tolist()
                f.write(json.dumps({"game": features, "result": r_night}) + "\n")

def write_data():
    raw_data = []
    i = 0
    with open("replay_buffer.json", "r") as f:
        for idx, line in enumerate(f):
            try:
                raw_data.append(json.loads(line))
            except json.JSONDecodeError:
                i+=1
                continue
    print("data read with", i, "errors after", idx, "lines")
    data = [(item["game"], item["result"]) for item in raw_data]
    dataset = GameValueDataset(data)
    
if __name__ == "__main__":
    self_play_and_generate_training_data()