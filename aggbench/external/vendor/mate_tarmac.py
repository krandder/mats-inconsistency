# TarMAC message aggregator vendored from https://github.com/XuehaiPan/mate/blob/3e631c0c3b043990fc53ae5fc3a37b0f65f230c5/examples/tarmac/models.py
# (class MessageAggregator, lines 15-48; MATE, Pan et al., NeurIPS 2022 Datasets and Benchmarks). License: MIT, see LICENSE-mate.
# This is a third-party reimplementation of TarMAC (Das et al. 2019, arXiv:1810.11187); the TarMAC authors
# did not release code. Only change: 'torch, nn = try_import_torch()' (RLlib) replaced by plain torch imports.
import numpy as np
import torch
from torch import nn


class MessageAggregator(nn.Module):
    def __init__(self, key_dim, value_dim, hidden_dim):
        super().__init__()

        self.key_dim = self.query_dim = key_dim
        self.value_dim = value_dim
        self.message_dim = self.key_dim + self.value_dim

        self.hidden_dim = hidden_dim

        self.query_predictor = nn.Linear(
            in_features=self.hidden_dim, out_features=self.query_dim, bias=False
        )
        self.scale = 1.0 / np.sqrt(self.query_dim)

    def forward(self, messages, hidden_states):
        # fmt: off
        assert messages.ndim == 3       # (B, T, Na * Dm)
        assert hidden_states.ndim == 3  # (B, T, Dh)
        B, T, joint_message_dim = messages.shape

        messages = messages.view(B, T, -1, self.message_dim)                   # (B, T, Na, Dm)
        keys, values = messages.split([self.key_dim, self.value_dim], dim=-1)  # (B, T, Na, *)

        queries = self.query_predictor(hidden_states)  # (B, T, Dq)
        queries = queries.unsqueeze(dim=-2)            # (B, T, 1, Dq)

        attns = self.scale * torch.matmul(queries, keys.transpose(-1, -2))  # (B, T, 1, Na)
        attns = attns.softmax(dim=-1)                                       # (B, T, 1, Na)
        outputs = torch.matmul(attns, values)                               # (B, T, 1, Dv)
        outputs = outputs.squeeze(dim=-2)                                   # (B, T, Dv)
        # fmt: on

        return outputs
