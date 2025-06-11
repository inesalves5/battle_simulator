import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import copy
import game
import main
from collections import Counter

class GameValueDataset(Dataset):
    def __init__(self, data):  # data = [(game, mode_reward), ...]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        game_state, reward = self.data[idx]
        return game_state.encode(), torch.tensor(reward, dtype=torch.float32)


def train_value_net(model, data, epochs=10, batch_size=10000, lr=1e-2, device='cpu'):
    model.to(device)
    dataset = GameValueDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model.forward(X)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def simulate_mode_reward(game_state, num_simulations):
    rewards = []
    games = [] 
    """
    objeivo do games é adicionar a data os varios estados do jogo que tiveram o resultado final r dentro do for,
    para serem considerados quando a nn receber um estado intermedio do jogo
    """
    for _ in range(num_simulations):
        g = copy.deepcopy(game_state)
        r = [0, 0]
        while not g.is_terminal():
            a0 = random.choice(g.actions_available(0))
            a1 = random.choice(g.actions_available(1))
            g, reward = g.get_next_state([a0, a1])
            if g is None:
                break
            r = [x+y for x, y in zip(r, reward)]
            games.append(g)
        if g is not None:
            r = [x+y for x, y in zip(r, g.reward_zone())]
            rewards.append(r[0])  # Only player 0's reward
    return rewards


def self_play_and_generate_training_data(n_games):
    data = []
    avg = 0
    options = 10
    values = []
    for _ in range(n_games):
        game_state = main.create_random_game()  # sua função que cria um estado inicial aleatório
        reward = simulate_mode_reward(game_state, options)  # usamos a moda em vez da média
        for res in reward:
            avg += res
            values.append(res)
            data.append((game_state, res))  # apenas o reward do jogador 0
    print("average reward:", avg / (n_games*options))
    counter = Counter(values)
    mode_value= counter.most_common(5)
    print("mode value:", mode_value)
    return data