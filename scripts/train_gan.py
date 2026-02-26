import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import get_dataloader, CheckerboardDataset
from src.model_gan import Generator, Discriminator
from src.metrics import compute_energy_distance, compute_wasserstein_distance

Z_DIM = 2        
LR = 1e-4        
BATCH_SIZE = 512
EPOCHS = 2000    
LOG_INTERVAL = 100 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    wandb.init(project="ML_HW4_GAN", name="GAN_Run_1")
    
    dataloader = get_dataloader(batch_size=BATCH_SIZE)
    eval_dataset = CheckerboardDataset(size=2000)
    real_data_eval = eval_dataset.data.clone().detach().to(dtype=torch.float32, device=DEVICE)

    G = Generator(z_dim=Z_DIM).to(DEVICE)
    D = Discriminator().to(DEVICE)
    
    optimizer_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    
    criterion = nn.BCEWithLogitsLoss()

    print(f"Start Training on {DEVICE}")

    for epoch in range(EPOCHS):
        total_loss_d = 0
        total_loss_g = 0
        
        for real_data in dataloader:
            real_data = real_data.to(DEVICE)
            batch_size = real_data.size(0)
            
            optimizer_D.zero_grad()
            
            label_real = torch.ones(batch_size, 1).to(DEVICE)
            output_real = D(real_data)
            loss_d_real = criterion(output_real, label_real)
            
            z = torch.randn(batch_size, Z_DIM).to(DEVICE)
            fake_data = G(z)
            label_fake = torch.zeros(batch_size, 1).to(DEVICE)
            output_fake = D(fake_data.detach())
            loss_d_fake = criterion(output_fake, label_fake)
            
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_D.step()
            total_loss_d += loss_d.item()

            optimizer_G.zero_grad()
            
            output_fake_for_g = D(fake_data) 
            loss_g = criterion(output_fake_for_g, label_real)
            
            loss_g.backward()
            optimizer_G.step()
            total_loss_g += loss_g.item()

        avg_loss_d = total_loss_d / len(dataloader)
        avg_loss_g = total_loss_g / len(dataloader)
        wandb.log({"loss_d": avg_loss_d, "loss_g": avg_loss_g}, step=epoch)
        
        if (epoch + 1) % LOG_INTERVAL == 0:
            with torch.no_grad():
                z_eval = torch.randn(2000, Z_DIM).to(DEVICE)
                fake_data_eval = G(z_eval)
                
                ed = compute_energy_distance(real_data_eval, fake_data_eval)
                wd = compute_wasserstein_distance(real_data_eval, fake_data_eval)
                
                print(f"Epoch [{epoch+1}/{EPOCHS}] Loss D: {avg_loss_d:.4f}, Loss G: {avg_loss_g:.4f} | ED: {ed:.4f}, WD: {wd:.4f}")
                
                wandb.log({"Energy Distance": ed, "Wasserstein Distance": wd}, step=epoch)
                
                plt.figure(figsize=(5,5))
                fake_np = fake_data_eval.cpu().numpy()
                plt.scatter(fake_np[:,0], fake_np[:,1], s=1, alpha=0.5)
                plt.xlim(-5, 5)
                plt.ylim(-5, 5)
                plt.title(f"GAN Epoch {epoch+1}")
                plt.savefig(f"gan_epoch_{epoch+1}.png")
                plt.close()

    torch.save(G.state_dict(), "generator.pth")
    torch.save(D.state_dict(), "discriminator.pth")
    print("Training Finished. Models saved.")
    wandb.finish()

if __name__ == "__main__":
    train()