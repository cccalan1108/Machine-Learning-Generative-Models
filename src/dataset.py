import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
class CheckerboardDataset(Dataset):
    def __init__(self, size=5000, root_dir='.'):
        self.size = size
        self.data = None
        possible_paths = [
            "checkerboard.npy",
            os.path.join("data", "checkerboard.npy"),
            os.path.join(os.path.dirname(__file__), "..", "checkerboard.npy"),
            os.path.join(os.path.dirname(__file__), "..", "data", "checkerboard.npy")
        ]


        target_path = None
        for path in possible_paths:
            if os.path.exists(path):
                target_path = path
                break

        if target_path:
            print(f"[Dataset] 成功: {target_path}")
            raw_data = np.load(target_path)
            self.data = torch.tensor(raw_data, dtype=torch.float32)
            self.size = len(self.data) 
        else:
            print("[Dataset] 錯誤：找不到 checkerboard.npy")
            scale = 4.0
            x1 = np.random.uniform(-scale, scale, size * 5)
            x2 = np.random.uniform(-scale, scale, size * 5)
            mask = (np.floor(x1) % 2 + np.floor(x2) % 2) % 2 == 0
            x1 = x1[mask][:size]
            x2 = x2[mask][:size]
            self.data = torch.tensor(np.stack([x1, x2], axis=1), dtype=torch.float32)
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return self.data[idx]

def get_dataloader(batch_size=128, size=5000):
    dataset = CheckerboardDataset(size=size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

