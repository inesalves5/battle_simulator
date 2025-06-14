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

def simulate(game_state):
    r = [0, 0]
    games = [] 
    """
    objeivo do games é adicionar a data os varios estados do jogo que tiveram o resultado final r dentro do for,
    para serem considerados quando a nn receber um estado intermedio do jogo
    """
    g = copy.deepcopy(game_state)
    while not g.is_terminal():
        games.append(g)
        a0 = random.choice(g.actions_available(0))
        a1 = random.choice(g.actions_available(1))
        g, reward = g.get_next_state([a0, a1])
        if g is None:
            break
        r = [x+y for x, y in zip(r, reward)]
    if g is not None:
        games.append(g)
        r = [x+y for x, y in zip(r, g.reward_zone())]
    return r[0], games


def self_play_and_generate_training_data(n_games):
    data = []
    values = []
    for _ in range(n_games):
        game_state = main.create_random_game()  # sua função que cria um estado inicial aleatório
        r, games = simulate(game_state)  # usamos a moda em vez da média
        for game in games:
            data.append((game, r))  # apenas o reward do jogador 0
    counter = Counter(data[i][1] for i in range(len(data)))
    mode_value= counter.most_common(5)
    print("mode value:", mode_value)
    return data