import tensorflow as tf
import numpy as np
import copy
import random
from collections import Counter
import main 
import NN
from itertools import product
import json

options = list(product(range(1, 11), repeat=2))

class GameValueDataset(tf.data.Dataset):
    def __new__(cls, data):
        # data: list of (game_state, reward)
        features = []
        targets = []
        for game_state, reward in data:
            features.append(game_state.numpy() if tf.is_tensor(game_state) else game_state)
            targets.append(reward)
        features = np.array(features, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        return tf.data.Dataset.from_tensor_slices((features, targets))

def train_value_net(epochs=100, batch_size=1000, lr=1e-2):
    data = []
    with open("res.json", "r") as f:
       for line in f:
            d = json.loads(line)
            features = d["game"]
            r = d["result"]
            data.append([features, r])
    dataset = GameValueDataset(data)
    dataset = dataset.shuffle(buffer_size=len(data)).batch(batch_size)
    model = NN.ValueMLP()
    try:
        model.load_weights('value_model_checkpoint.weights.h5')
    except Exception as e:
        print("Error loading weights:", e)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse')
    model.fit(dataset, epochs=epochs)
    model.save_weights('value_model_checkpoint.weights.h5')
    return model

def simulate(game_state):
    r = [0, 0]
    games = []
    g = copy.deepcopy(game_state)
    while not g.is_terminal():
        games.append(g)
        a0 = g.action_available(0)
        a1 = g.action_available(1)
        g, reward = g.get_next_state([a0, a1])
        if g is None:
            break
        r = [x + y for x, y in zip(r, reward)]
    if g is not None and g not in games:
        games.append(g)
        r = [x + y for x, y in zip(r, g.reward_zone())]
    elif g is None:
        return None, None
    return r[0], games

def self_play_and_generate_training_data():
    print("options is:", len(options))
    with open("res.json", "a") as f:
        for n in options:
            a, j = n
            print("-----------------------case", n)
            game_state = main.create_random_game("day", a, j)
            r, games = simulate(game_state)
            if r is None:
                continue
            print("final reward in simulate:", r)
            all_games = []
            for game_part in games:
                all_games += game_part.generate_equivalent_games()
            for g in all_games:
                features = g.encode().numpy().tolist()
                f.write(json.dumps({"game": features, "result": r}) + "\n")

if __name__ == "__main__":
    with open("results.txt", "a") as f:
        for _ in range(100):
            self_play_and_generate_training_data(f)
"""
if __name__ == "__main__":
    with open("results.txt", "r") as f:
        train_value_net(f)
"""