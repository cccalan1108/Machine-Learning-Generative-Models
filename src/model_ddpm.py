import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),  
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return x + self.block(x)

class NoisePredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=512, time_dim=128, num_layers=6):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.t_proj = nn.Linear(time_dim, hidden_dim)
        
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout=0.0) for _ in range(num_layers)
        ])
        
        self.final_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        t_hidden = self.t_proj(t_emb)
        h = self.input_proj(x) + t_hidden
        
        for block in self.res_blocks:
            h = block(h)
            
        return self.final_mlp(h)