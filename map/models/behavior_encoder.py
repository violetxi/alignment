import torch
import numpy as np
import torch.nn.functional as F

from typing import Dict
from torch import nn
from torch.optim import Adam
from tianshou.data import to_torch, Batch

class MeanEncoder(nn.Module):
    def __init__(self, num_agent, layer_num, state_shape, history_length, hidden_units=128, embedding_dim=8, device='cpu'):
        super().__init__()
        self.device = device
        # plus one for the action
        self.model = [
            nn.Linear((np.prod(state_shape) + 1)*history_length, hidden_units),
            nn.ReLU()]
        for i in range(layer_num - 1):
            self.model += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        self.model += [nn.Linear(hidden_units, embedding_dim)]
        self.num_agent = num_agent
        self.model = nn.Sequential(*self.model)
        self.optim = Adam(self.model.parameters(), lr=1e-3)
        self.embedding_dim = embedding_dim

    def forward(self, s, **kwargs):
        s = to_torch(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits
    
class CovariateEncoder(nn.Module):
    def __init__(self, num_agent, layer_num, state_shape, history_length, hidden_units=128, embedding_dim=8, device='cpu'):
        super().__init__()
        self.device = device
        # plus one for the action
        self.model = [
            nn.Linear((np.prod(state_shape) + 1)*history_length, hidden_units),
            nn.ReLU()]
        for i in range(layer_num - 1):
            self.model += [nn.Linear(hidden_units, hidden_units), nn.ReLU()]
        self.model += [nn.Linear(hidden_units, embedding_dim), nn.ReLU()]
        self.num_agent = num_agent
        self.model = nn.Sequential(*self.model)
        self.optim = Adam(self.model.parameters(), lr=1e-3)

    def forward(self, s, **kwargs):
        s = to_torch(s, device=self.device, dtype=torch.float)
        batch = s.shape[0]
        s = s.view(batch, -1)
        logits = self.model(s)
        return logits

