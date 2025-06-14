import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import copy
from game import Game

TOTAL_UNITS = 4

class Transformer(nn.Module):
    def __init__(self, input_dim=TOTAL_UNITS, embed_dim=64, nhead=4, nlayers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        value = self.value_head(pooled)
        return value.squeeze(-1)  # shape: (batch,)

    def predict(self, game_state):
        self.eval()
        with torch.no_grad():
            if not isinstance(game_state, torch.Tensor):
                game_state = torch.tensor(game_state, dtype=torch.float32)
            game_state = game_state.unsqueeze(0).to(next(self.parameters()).device)  # add batch dim
            value = self.forward(game_state)
            return value.item()

class ValueMLP(nn.Module):
    def __init__(self, input_dim=TOTAL_UNITS):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def predict(self, game_state):
        with torch.no_grad():
            x = game_state.encode().unsqueeze(0)  # add batch dim
            return self.forward(x).item()
