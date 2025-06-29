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

def train_value_net(model, data, epochs=10, batch_size=10000, lr=1e-2):
    dataset = GameValueDataset(data)
    dataset = dataset.shuffle(buffer_size=len(data)).batch(batch_size)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse')

    model.fit(dataset, epochs=epochs)

def simulate(game_state):
    r = [0, 0]
    games = []
    g = copy.deepcopy(game_state)
    while not g.is_terminal():
        games.append(g)
        a0 = random.choice(g.actions_available(0))
        a1 = random.choice(g.actions_available(1))
        g, reward = g.get_next_state([a0, a1])
        if g is None:
            break
        r = [x + y for x, y in zip(r, reward)]
    if g is not None:
        games.append(g)
        r = [x + y for x, y in zip(r, g.reward_zone())]
    return r[0], games

def self_play_and_generate_training_data(n_games):
    data = []
    for _ in range(n_games):
        game_state = main.create_random_game()
        r, games = simulate(game_state)
        for game in games:
            data.append((game, float(r)))  # Convert reward para float

    counter = Counter([reward for _, reward in data])
    mode_value = counter.most_common(5)
    print('mode:', mode_value)
    return data
