import tensorflow as tf
import numpy as np
import copy
import random
from collections import Counter
import main 

class GameValueDataset(tf.data.Dataset):
    def __new__(cls, data):
        # data: list of (game_state, reward)
        features = []
        targets = []
        for game_state, reward in data:
            features.append(game_state.encode().numpy() if tf.is_tensor(game_state.encode()) else game_state.encode())
            targets.append(reward)
        features = np.array(features, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        return tf.data.Dataset.from_tensor_slices((features, targets))

def train_value_net(model, data, epochs=50, batch_size=1000, lr=1e-2):
    dataset = GameValueDataset(data)
    dataset = dataset.shuffle(buffer_size=len(data)).batch(batch_size)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse')

    model.fit(dataset, epochs=epochs)

def simulate(game_state):
    r = [0, 0]
    games = []
    g = copy.deepcopy(game_state)
    while g.is_terminal():
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
    return r[0], games

def self_play_and_generate_training_data(file):
    with open(file, "w") as f:
        #for _ in range(100):
        game_state = main.create_random_game()
        r, games = simulate(game_state)
        print("sim ended")
        """
        if game_state not in games:
            games.append(game_state)
        all_games = games.copy()
        for game_part in games:
            equivalent = game_part.generate_equivalent_games()
            all_games.append(game_ex for game_ex in equivalent if game_ex not in all_games)
        for game_state in all_games:
            file.write(f"{game_state.encode()}:{r}\n")
        """
        equivalent_games = game_state.generate_equivalent_games()
        print("eq genrerated")
        equivalent_games.append(game_state)
        for game_ex in equivalent_games:
            f.write(f"{game_ex.encode()}:{r}\n")
        print("games written")
    return list(zip(equivalent_games, [r] * len(equivalent_games)))

if __name__ == "__main__":
    with open("results.txt", "a") as f:
        for _ in range(100):
            self_play_and_generate_training_data(f)
"""
if __name__ == "__main__":
    with open("results.txt", "r") as f:
        train_value_net(f)
"""