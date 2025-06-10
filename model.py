# file: model.py

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

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t):
        h = self.relu(self.conv1(x))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        h = self.relu(self.conv2(h))
        return h

class UNet(nn.Module):
    def __init__(self, in_channels=4, time_emb_dim=32): # 3 for image + 1 for map
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Encoder
        self.down1 = Block(in_channels, 64, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Block(64, 128, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bot = Block(128, 256, time_emb_dim)
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up1 = Block(256, 128, time_emb_dim)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up2 = Block(128, 64, time_emb_dim)
        
        # Output
        self.out = nn.Conv2d(64, 1, 1) # Output 1 channel logit map

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        x1 = self.down1(x, t_emb)
        p1 = self.pool1(x1)
        
        x2 = self.down2(p1, t_emb)
        p2 = self.pool2(x2)
        
        bot = self.bot(p2, t_emb)
        
        up1 = self.upconv1(bot)
        up1 = torch.cat([up1, x2], dim=1)
        up1 = self.up1(up1, t_emb)
        
        up2 = self.upconv2(up1)
        up2 = torch.cat([up2, x1], dim=1)
        up2 = self.up2(up2, t_emb)
        
        return self.out(up2)
