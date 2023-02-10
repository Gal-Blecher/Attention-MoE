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
        self.warm_up = 0

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(1)
        attn = F.softmax(score, dim=0)
        # if self.warm_up < 50:
        #     self.warm_up += 1
        #     attn = torch.full_like(attn, 0.5)
        return attn