import torch.nn as nn
import torch
from typing import Optional, Tuple
import torch.nn.functional as F

class AdditiveAttention(nn.Module):

    def __init__(self, input_dim) -> None:
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.key_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(input_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(input_dim, 1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(1)
        attn = F.softmax(score, dim=0)
        return attn