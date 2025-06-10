# file: dataset.py

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import scipy.io
import numpy as np

class ShanghaiTechDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=(256, 256)):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        if split == 'train':
            self.img_dir = os.path.join(root_dir, 'part_A_final', 'train_data', 'images')
            self.gt_dir = os.path.join(root_dir, 'part_A_final', 'train_data', 'ground-truth')
        else: # 'test'
            self.img_dir = os.path.join(root_dir, 'part_A_final', 'test_data', 'images')
            self.gt_dir = os.path.join(root_dir, 'part_A_final', 'test_data', 'ground-truth')
            
        self.img_files = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        original_w, original_h = image.size
        
        # Apply transformations
        image_tensor = self.transform(image)
        
        # Load ground truth
        gt_name = 'GT_' + os.path.splitext(img_name)[0] + '.mat'
        gt_path = os.path.join(self.gt_dir, gt_name)
        mat = scipy.io.loadmat(gt_path)
        gt_points = mat['image_info'][0, 0][0, 0][0]
        
        # Create the ground truth binary map x_0
        x_0 = torch.zeros(self.img_size, dtype=torch.float32)
        
        # Scale coordinates and place points
        for point in gt_points:
            x, y = point
            # Scale coordinates to the new image size
            new_x = int(x * (self.img_size[1] / original_w))
            new_y = int(y * (self.img_size[0] / original_h))
            
            # Clamp to be within bounds
            new_x = min(max(0, new_x), self.img_size[1] - 1)
            new_y = min(max(0, new_y), self.img_size[0] - 1)
            
            x_0[new_y, new_x] = 1.0
            
        return image_tensor, x_0
