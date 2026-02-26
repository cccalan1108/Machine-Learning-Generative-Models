import sys
import os
import torch
import torch.optim as optim
import wandb
import imageio
import numpy as np
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import get_dataloader, CheckerboardDataset
from src.model_ddpm import NoisePredictor
from src.diffusion import DDPMScheduler
from src.metrics import compute_energy_distance, compute_wasserstein_distance

BATCH_SIZE = 512
LR = 2e-4
EPOCHS = 3000   
LOG_INTERVAL = 100
EMA_DECAY = 0.999
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
    
    def update(self, model):
        with torch.no_grad():
            for s_param, param in zip(self.shadow.parameters(), model.parameters()):
                if param.requires_grad:
                    s_param.data = s_param.data * self.decay + param.data * (1.0 - self.decay)

    def get_model(self):
        return self.shadow

def train():
    wandb.init(project="ML_HW4_DDPM", name="DDPM_Linear_LayerNorm")
    
    dataloader = get_dataloader(batch_size=BATCH_SIZE)
    
    eval_ds = CheckerboardDataset()
    if eval_ds.data is None: 
        print("Error: Dataset empty")
        return
    real_data_eval = eval_ds.data.clone().to(DEVICE)

    model = NoisePredictor(hidden_dim=512, num_layers=4).to(DEVICE)
    ema = EMA(model, EMA_DECAY)
    
    scheduler = DDPMScheduler(num_timesteps=1000, schedule='linear', device=DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()
    
    best_ed = float('inf')
    frames = []

    print(f"開始訓練 (Device: {DEVICE}, Schedule: Cosine)")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for x_start in dataloader:
            x_start = x_start.to(DEVICE)
            t = torch.randint(0, scheduler.num_timesteps, (x_start.shape[0],), device=DEVICE).long()
            x_t, noise = scheduler.add_noise(x_start, t)
            noise_pred = model(x_t, t)
            loss = loss_fn(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            total_loss += loss.item()

        if (epoch + 1) % LOG_INTERVAL == 0:
            avg_loss = total_loss / len(dataloader)
            ema_model = ema.get_model()
            
            samples = scheduler.ddim_sample(ema_model, shape=(5000, 2), ddim_timesteps=100, eta=0.0)
            
            ed = compute_energy_distance(real_data_eval, samples)
            wd = compute_wasserstein_distance(real_data_eval, samples)
            
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | ED: {ed:.6f} | WD: {wd:.6f}")
            wandb.log({"loss": avg_loss, "ED": ed, "WD": wd}, step=epoch)
            
            if ed < best_ed:
                best_ed = ed
                torch.save(ema_model.state_dict(), "ddpm_best_ema.pth")
                print(f"New Best Model Saved (ED: {best_ed:.6f})")

            plt.figure(figsize=(5,5))
            plt.scatter(samples.cpu().numpy()[:,0], samples.cpu().numpy()[:,1], s=1, alpha=0.5, c='blue')
            plt.xlim(-5, 5); plt.ylim(-5, 5)
            plt.savefig(f"epoch_{epoch+1}.png")
            plt.close()
            frames.append(f"epoch_{epoch+1}.png")

    torch.save(ema.get_model().state_dict(), "ddpm_final_ema.pth")
    print("Training Complete.")
    wandb.finish()

if __name__ == "__main__":
    train()