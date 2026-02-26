import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class DDPMScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule='linear', device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
            
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x_start, t):
        noise = torch.randn_like(x_start)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        x_t = sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise
        return x_t, noise

    @torch.no_grad()
    def sample(self, model, shape):
        model.eval()
        x = torch.randn(shape, device=self.device)
        for t in tqdm(reversed(range(self.num_timesteps)), desc="DDPM Sampling", total=self.num_timesteps, leave=False):
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            predicted_noise = model(x, t_tensor)
            
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            
            coef = beta_t / torch.sqrt(1 - alpha_cumprod_t)
            mean = (1 / torch.sqrt(alpha_t)) * (x - coef * predicted_noise)
            
            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * noise
            else:
                x = mean
        return x

    @torch.no_grad()
    def ddim_sample(self, model, shape, ddim_timesteps=50, eta=0.0):
        model.eval()
        x = torch.randn(shape, device=self.device)
        
        c = self.num_timesteps // ddim_timesteps
        seq = np.linspace(0, self.num_timesteps - 1, ddim_timesteps, dtype=int)
        seq = list(reversed(seq))
        seq_next = seq[1:] + [-1]
        
        for i, t in enumerate(tqdm(seq, desc=f"DDIM (steps={ddim_timesteps})", leave=False)):
            t_next = seq_next[i]
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            
            epsilon_theta = model(x, t_tensor)
            
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_t_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0).to(self.device)
            
            sigma_t = eta * torch.sqrt((1 - alpha_bar_t_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_next))
            
            pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * epsilon_theta) / torch.sqrt(alpha_bar_t)
            dir_xt = torch.sqrt(1 - alpha_bar_t_next - sigma_t**2) * epsilon_theta
            
            noise = torch.randn_like(x)
            x = torch.sqrt(alpha_bar_t_next) * pred_x0 + dir_xt + sigma_t * noise
            
        return x