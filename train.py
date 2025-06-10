# file: train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import os

from dataset import ShanghaiTechDataset
from model import UNet

# --- Hyperparameters ---
DATA_ROOT = './data/shanghaitech'
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
TIMESTEPS = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = './checkpoints'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- 1. Forward Noising Process ---
def get_noising_schedule(timesteps, initial_ones, final_ones):
    return torch.linspace(initial_ones, final_ones, timesteps + 1, device=DEVICE)

def forward_noising(x_0, t, schedule, H, W):
    """Generates a noisy map x_t from a clean map x_0."""
    num_gt_ones = x_0.sum().long()
    
    # Use schedule for the *total* number of ones
    total_pixels = H * W
    n_t_schedule = torch.round(
        get_noising_schedule(TIMESTEPS, 0, total_pixels) + num_gt_ones.clamp(max=total_pixels)
    ).long().clamp(max=total_pixels)

    num_target_ones = n_t_schedule[t].item()
    num_to_add = num_target_ones - num_gt_ones
    
    if num_to_add <= 0:
        return x_0.clone()

    x_t = x_0.clone()
    zero_coords = (x_t == 0).nonzero()
    
    if len(zero_coords) == 0:
        return x_t # Already full

    num_to_add = min(num_to_add, len(zero_coords))
    indices_to_flip = torch.randperm(len(zero_coords))[:num_to_add]
    coords_to_flip = zero_coords[indices_to_flip]
    
    x_t[coords_to_flip[:, 0], coords_to_flip[:, 1]] = 1
    return x_t

# --- 2. Setup Dataloader, Model, Optimizer ---
print(f"Using device: {DEVICE}")

train_dataset = ShanghaiTechDataset(root_dir=DATA_ROOT, split='train', img_size=IMG_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

model = UNet(in_channels=4).to(DEVICE) # 3 for image, 1 for noisy map
optimizer = Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss() # Numerically stable

# --- 3. Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for i, (image, x_0) in enumerate(progress_bar):
        optimizer.zero_grad()
        
        image = image.to(DEVICE)
        x_0 = x_0.to(DEVICE)
        batch_size = image.shape[0]

        # 1. Sample random timesteps for each item in the batch
        t = torch.randint(1, TIMESTEPS + 1, (batch_size,), device=DEVICE)

        # 2. Generate noisy maps x_t for the batch
        # We need to loop because each image has a different number of initial points (num_gt_ones)
        x_t_list = [forward_noising(x_0[i], t[i], None, IMG_SIZE[0], IMG_SIZE[1]) for i in range(batch_size)]
        x_t = torch.stack(x_t_list)

        # 3. Prepare model input
        # Add channel dimension to x_t
        model_input = torch.cat([image, x_t.unsqueeze(1)], dim=1)

        # 4. Get model's prediction of the clean map
        predicted_x_0_logits = model(model_input, t.float()) # Pass t as float
        
        # 5. Calculate loss
        # Squeeze channel dim from prediction to match x_0 shape
        loss = criterion(predicted_x_0_logits.squeeze(1), x_0)
        
        # 6. Backpropagate and update
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Average Loss: {avg_loss:.4f}")

    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'point_diffusion_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

print("Training finished.")
