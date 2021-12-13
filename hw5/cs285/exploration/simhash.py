from __future__ import annotations

from collections import defaultdict
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import infrastructure.pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel


class SimHashExplorationModel(BaseExplorationModel, nn.Module):

    def __init__(self, state_dim: int | Tuple[int, ...], granularity: int, preprocess_fn: torch.Module = None) -> None:
        super().__init__()
        if type(state_dim) == int:
            state_dim = (state_dim, )
        self.preprocess = preprocess_fn or nn.Identity()
        self.A = torch.normal(0, 1, (granularity, *state_dim)).to(ptu.device)
        self.hashtable = defaultdict(lambda: 1)
        self.state_dim: Tuple[int, ...] = state_dim

    def forward(self, obs):
        hashes = self.hash(obs).cpu().numpy()
        bonuses = torch.zeros((obs.shape[0],))
        for i in range(obs.shape[0]):
            hash = tuple(hashes[i].tolist())
            bonuses[i] += 1.0 / np.sqrt(self.hashtable[hash])
        return bonuses.to(ptu.device)

    def forward_np(self, obs):
        return self.forward(obs).cpu().numpy()

    def hash(self, state):
        return torch.sign(self.A @ self.preprocess(state).T).T

    def update(self, ob_no, ac_na, next_ob_no, re_n, terminal_n):
        hashes = self.hash(ob_no).to('cpu').numpy()
        for i in range(ob_no.shape[0]):
            hash = tuple(hashes[i].tolist())
            self.hashtable[hash] += 1
        return 0