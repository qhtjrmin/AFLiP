import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MLPLinkDecoder(nn.Module):
    """Hadamard product → MLP → sigmoid (original behavior preserved)."""
    def __init__(self, in_channels: int, hidden_dims: int, out_channels: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers == 1:
            self.lins = nn.ModuleList([nn.Linear(in_channels, out_channels)])
        else:
            self.lins = nn.ModuleList([nn.Linear(in_channels, hidden_dims)])
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_dims, hidden_dims))
            self.lins.append(nn.Linear(hidden_dims, out_channels))
        self.dropout = float(dropout)

    def reset_parameters(self) -> None:
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = F.relu(lin(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class DotProductLinkDecoder(nn.Module):
    """Dot product → raw logit (same output format as MLPLinkDecoder)."""
    def __init__(self):
        super().__init__()

    def reset_parameters(self) -> None:
        pass  # no parameters to reset

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        # x_i, x_j: [B, d]
        # output:   [B, 1] (raw logit)
        logit = (x_i * x_j).sum(dim=-1, keepdim=True)
        return logit
