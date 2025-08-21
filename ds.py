import tensorflow as tf
import numpy as np
import copy
import random
from collections import Counter
import main 
import NN
from itertools import product
import json
import fully_connected

options = list(product(range(1, 11), repeat=2))

class GameValueDataset(tf.data.Dataset):
    def __new__(cls, data):
        # data: list of (game_state, reward)
        features = []
        targets = []
        for (game_state, reward) in data:
            features.append(game_state.numpy() if tf.is_tensor(game_state) else game_state)
            targets.append(reward)
        features = np.memmap("features.dat", dtype=np.float16, mode="w+", shape=(3562344, 255))
        targets = np.memmap("targets.dat", dtype=np.float16, mode="w+", shape=(3562344,))
        return tf.data.Dataset.from_tensor_slices((features, targets))

class PredictAfterEpoch(tf.keras.callbacks.Callback):
    def __init__(self, test_inputs="test_cases_encoded.json", log_path="/fully_connected/test_cases.txt"):
        super().__init__()
        self.test_inputs = [l.strip() for l in open(test_inputs)]  # shape: (n_samples, n_features)
        self.log_path = log_path

    def on_epoch_end(self, epoch, logs=None):
        for i in range(len(self.test_inputs)):
            prediction = self.model.predict(self.test_inputs[i], verbose=0).flatten()
            with open(self.log_path, "a") as f:
                json.dump({"epoch": epoch + 1, "sample": i, "prediction": float(prediction)}, f)
                f.write("\n")

def train_value_net(epochs=10, batch_size=64, lr=1e-3):
    data = []
    with open("replay_buffer.json", "r") as f:
        raw_data = [json.loads(line) for line in f]

    # transformar em lista de (features, result)
    data = [(item["game"], item["result"]) for item in raw_data]
    dataset = GameValueDataset(data)
    dataset = dataset.shuffle(buffer_size=len(data)).batch(batch_size)
    model = fully_connected.FullyConnectedNetwork()
    try:
        model.load_weights('fully_connected.weights.h5')
    except Exception as e:
        print("")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse')
    callbacks = [PredictAfterEpoch()]
    history = model.fit(dataset, epochs=epochs, callbacks=callbacks)
    with open("/fully_connected/training.json", "w") as file:
        json.dump(history.history, file)
    model.save_weights('fully_connected.weights.h5')
    return model

def simulate(game_state):
    r = 0
    buffer = []
    g = copy.deepcopy(game_state)
    while not g.is_terminal():
        a0 = g.action_available(0)
        a1 = g.action_available(1)
        buffer.append((a0, g))
        g, reward, _ = g.get_next_state([a0, a1])
        if g is None:
            return None, None
        r += reward[0]
    buffer.append((None, g))
    r += g.reward_zone()[0]
    return r, buffer
    

def self_play_and_generate_training_data():
    with open("replay_buffer.json", "a") as f:
        for n in options:
            a, j = n
            print("-----------------------case", n)
            game_state = main.create_random_game("day", a, j)
            r, buffer = simulate(game_state)
            if r is None:
                continue
            print("result is:", r)
            all_games = []
            for (a0, game_part) in buffer:
                all_games += game_part.generate_equivalent_games(a0)
            for (a0, g) in all_games:
                features = g.encode(a0).numpy().tolist()
                f.write(json.dumps({"game": features, "result": r}) + "\n")

if __name__ == "__main__":
    train_value_net()