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
import mcts
import os
import shutil
import graphs

options = list(product(range(1, 6), repeat=2))
INPUT_DIM = 190+64+1 # 190 for state, 64 for action, 1 for action type
models = {
    "resnet_base": lambda: resnet.ResidualNetwork(INPUT_DIM),
    "fully_connected_base": lambda: fully_connected.FullyConnectedNetwork(INPUT_DIM),
    "resnet_larger": lambda: resnet.ResidualNetwork_Larger(INPUT_DIM),
    "fully_connected_larger": lambda: fully_connected.FullyConnectedNetwork_Larger(INPUT_DIM),
    "resnet_deeper": lambda: resnet.ResidualNetwork_Deeper(INPUT_DIM),
    "fully_connected_deeper": lambda: fully_connected.FullyConnectedNetwork_Deeper(INPUT_DIM),
}

"""def serialize_example(feature, target):
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

def update_tfrecord(new_data, old_filename="dataset.tfrecord"):
    old_examples = load_dataset(old_filename)
    remove_count = len(new_data)
    remaining_examples = old_examples.skip(remove_count)
    
    serialized_examples = [serialize_example(f, t) for f, t in new_data]

    new_dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(serialized_examples, dtype=tf.string)
    )
    new_dataset = new_dataset.map(lambda x: parse_example(x))

    updated_examples = remaining_examples.concatenate(new_dataset)
    final_data = []
    for features, target in updated_examples.unbatch():
        final_data.append((features, target))
    write_tfrecord(final_data)
    print("TFRecord atualizado")

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
"""
def load_dataset(nn_type, num_features=255, batch_size=64):
    filename=f"dataset_{nn_type}.tfrecord"
    raw_dataset = tf.data.TFRecordDataset(filename)
    dataset = raw_dataset.map(lambda x: parse_example(x, num_features))
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

class GameValueDataset(tf.data.Dataset):
    def __new__(cls, nn_type, write=False, update=False):
        """
        if write:
            return write_tfrecord(data)
        if update:
            return update_tfrecord(data)
        """
        return load_dataset(nn_type)

class SaveLast2Weights(tf.keras.callbacks.Callback):
        def __init__(self, nn_type):
            self.dir_path = f"{nn_type}"

        def on_epoch_end(self, epoch, logs=None):
            # guardar pesos
            if os.path.exists(self.file1):
                shutil.copy(self.file1, self.file2)

            self.model.save_weights(self.file1)

            with open(f"{self.dir_path}/loss.json", "a") as f:
                f.write(f"{logs['loss']}\n")

class UpdateReplayBuffer(tf.keras.callbacks.Callback):
    def __init__(self, nn_type):
        super().__init__()
        self.nn_type = nn_type
        self.old_filename = f"dataset_{nn_type}.tfrecord"

    def on_epoch_end(self, epoch, logs=None):
        self_play_and_generate_training_data(self.nn_type, epoch=epoch)

class PredictAfterEpoch(tf.keras.callbacks.Callback):
    def __init__(self, nn_type, current_epoch=0, test_inputs="test_cases_encoded.json"):  # logpath depends on nn type
        super().__init__()
        self.test_inputs = [l.strip() for l in open(test_inputs)]  # shape: (n_samples, n_features)
        self.log_path = f"{nn_type}/test_cases.json"
        self.current_epoch = current_epoch

    def on_epoch_end(self, epoch, logs=None):
        epoch = self.current_epoch + 1
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

def train_value_net(model, nn_type, epochs=100, batch_size=64, lr=1e-4):
    dataset = GameValueDataset(nn_type)
    dataset = dataset.map(lambda x, y: (tf.reshape(x, [tf.shape(x)[0], 255]), y))
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))
    dummy_input = tf.zeros((1, INPUT_DIM))
    _ = model(dummy_input)  # constrói o modelo    
    try:
        model.load_weights(f"{nn_type}.weights.h5") # depends on nn type
    except Exception as e:
        print("erro:", e)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse')
    callbacks = [PredictAfterEpoch(nn_type), SaveLast2Weights(nn_type), UpdateReplayBuffer(nn_type)]
    history = model.fit(dataset, epochs=epochs, callbacks=callbacks)
    return model

    """
    data = []  # para quando nao esta a ser escrito
    dataset = GameValueDataset(data)
    dataset = dataset.map(lambda x, y: (tf.reshape(x, [tf.shape(x)[0], 255]), y))
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))

    dummy_input = tf.zeros((1, INPUT_DIM))
    _ = model(dummy_input)  # constrói o modelo    
    
    try:
        model.load_weights(f"{nn_type}.weights.h5")  # depende do tipo de rede
        print("Pesos carregados.")
    except Exception as e:
        print("erro ao carregar pesos:", e)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mse')

    history_all = {"loss": []}
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        callbacks = [PredictAfterEpoch(nn_type, current_epoch=epoch)]
        history = model.fit(dataset, epochs=1, callbacks=callbacks, verbose=1)
        history_all["loss"].append(history.history["loss"][0])
        with open(f"{nn_type}/training.json", "a", encoding="utf-8") as file:
            json.dump(history_all, file)

        # salva pesos após cada epoch
        model.save_weights(f"{nn_type}.weights.h5")
    return model
    """

def simulate_random(game_state, iterations, action):
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

def simulate_from_mcts(game_state, iterations, action, my_nn=None, opponent_nn=None):
    r = 0
    buffer = []
    iterations = iterations * 50
    g = copy.deepcopy(game_state)

    while not g.is_terminal():
        max_reward = g.max_reward(action)

        root_jap = mcts.DecisionNode(copy.deepcopy(g), max_reward=max_reward, player=0, root=True)
        tree_jap = mcts.MCTS(root_jap, nn=my_nn)
        node_jap = tree_jap.search(root_jap, iterations=iterations)
        action_jap = node_jap.action

        root_all = mcts.DecisionNode(copy.deepcopy(g), max_reward=max_reward, player=1, root=True)
        tree_all = mcts.MCTS(root_all, nn=opponent_nn)
        node_all = tree_all.search(root_all, iterations=iterations)
        action_all = node_all.a_action
        buffer.append((action_jap, copy.deepcopy(g)))

        g, reward, rolls = g.get_next_state([action_jap, action_all])
        r += reward[0]
    buffer.append((None, copy.deepcopy(g)))
    r += copy.deepcopy(g).reward_zone()[0]
    return r, buffer

def self_play_and_generate_training_data(nn_type, epoch=0):
    if epoch < 2:
        my_nn, opponent_nn = None, None
    else:
        my_nn = models[nn_type]()  # create fresh model
        my_nn.load_weights(f"{nn_type}_1.weights.h5") # depends on nn type
        opponent_nn = models[nn_type]()  # create fresh model
        opponent_nn.load_weights(f"{nn_type}_2.weights.h5") # depends on nn

    writer = tf.io.TFRecordWriter(f"dataset_{nn_type}.tfrecord")
    for _ in range(4): #100 jogos de cada vez
        for n in options:
            a, j = n
            print("-----------------------case", n)

            game_state_day, game_state_night = main.create_random_game_pair(a, j)

            r_day, buffer_day = simulate_from_mcts(
                copy.deepcopy(game_state_day), iterations=a+j, action="day",
                my_nn=my_nn, opponent_nn=opponent_nn)
            if r_day is None:
                continue

            r_night, buffer_night = simulate_from_mcts(
                copy.deepcopy(game_state_night), iterations=a+j, action="night",
                my_nn=my_nn, opponent_nn=opponent_nn)    
            if r_night is None:
                continue

            all_games_day, all_games_night = [], []
            for (a0, game_part) in buffer_day:
                all_games_day += game_part.generate_equivalent_games(a0)
            for (a0, g) in all_games_day:
                features = g.encode(a0).numpy().tolist()
                example = tf.train.Example(
                    features=tf.train.Features(feature={
                        "game": tf.train.Feature(float_list=tf.train.FloatList(value=features)),
                        "result": tf.train.Feature(float_list=tf.train.FloatList(value=[r_day]))
                    })
                )
                writer.write(example.SerializeToString())

            for (a0, game_part) in buffer_night:
                all_games_night += game_part.generate_equivalent_games(a0)
            for (a0, g) in all_games_night:
                features = g.encode(a0).numpy().tolist()
                example = tf.train.Example(
                    features=tf.train.Features(feature={
                        "game": tf.train.Feature(float_list=tf.train.FloatList(value=features)),
                        "result": tf.train.Feature(float_list=tf.train.FloatList(value=[r_night]))
                    })
                )
                writer.write(example.SerializeToString())
    writer.close()
    _trim_tfrecord(nn_type)


def _trim_tfrecord(nn_type, max_lines=3000000):
    filename=f"dataset_{nn_type}.tfrecord"
    raw_dataset = tf.data.TFRecordDataset(filename)
    all_examples = list(raw_dataset)

    if len(all_examples) > max_lines:
        with tf.io.TFRecordWriter(filename) as writer:
            for ex in all_examples[-max_lines:]:
                writer.write(ex.numpy())

    """
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) > max_lines:
        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(lines[-max_lines:])
    
    with open("replay_buffer.json", "a", encoding="utf-8") as f:
        for n in options:
            a, j = n
            print("-----------------------case", n)
            game_state_day, game_state_night = main.create_random_game_pair(a, j)
            r_day, buffer_day = simulate_from_mcts(copy.deepcopy(game_state_day), iterations=a+j, action="day", my_nn=my_nn, opponent_nn=opponent_nn)
            if r_day is None:
                continue
            r_night, buffer_night = simulate_from_mcts(copy.deepcopy(game_state_night), iterations=a+j, action="night", my_nn=my_nn, opponent_nn=opponent_nn)
            if r_night is None:
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
def write_data(x=None):
    raw_data = []
    i = 0
    with open("replay_buffer.json", "r") as f:
        for idx, line in enumerate(f):
            try:
                raw_data.append(json.loads(line))
            except json.JSONDecodeError:
                i+=1
                continue
    print(f"Linhas inválidas ignoradas: {i}")
    if x is not None:
        data_to_write = raw_data[x:]
        raw_data = raw_data[:x]

        with open("replay_buffer.json", "w") as f:
            for item in data_to_write:
                f.write(json.dumps({"game": item["game"], "result": item["result"]}) + "\n")

    data = [(item["game"], item["result"]) for item in raw_data]
    dataset = GameValueDataset(data, write=True if x is None else False, update=True if x is not None else False)

def train_and_save(key):
    model = models[key]()  # create fresh model
    train_value_net(model=model, nn_type=key)
    print(f"Modelo {key} terminado!")
    """
if __name__ == "__main__":
    nn_type = "fully_connected_base"  
    self_play_and_generate_training_data(nn_type)
    train_and_save(nn_type)
    graphs.run_all_plots(nn_type)
    print("Tudo terminado para", nn_type)
