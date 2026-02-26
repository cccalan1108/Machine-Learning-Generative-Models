import torch
import ot
import numpy as np

def compute_energy_distance(x_real, x_fake):
    if isinstance(x_real, torch.Tensor): x_real = x_real.detach().cpu()
    if isinstance(x_fake, torch.Tensor): x_fake = x_fake.detach().cpu()
    dist_xy = torch.cdist(x_real, x_fake, p=2)
    dist_xx = torch.cdist(x_real, x_real, p=2)
    dist_yy = torch.cdist(x_fake, x_fake, p=2)
    
    ed = 2 * torch.mean(dist_xy) - torch.mean(dist_xx) - torch.mean(dist_yy)
    return ed.item()

def compute_wasserstein_distance(x_real, x_fake):
    if isinstance(x_real, torch.Tensor): x_real = x_real.detach().cpu().numpy()
    if isinstance(x_fake, torch.Tensor): x_fake = x_fake.detach().cpu().numpy()

    n = x_real.shape[0]
    m = x_fake.shape[0]
    a = np.ones((n,)) / n
    b = np.ones((m,)) / m 
    M = ot.dist(x_real, x_fake, metric='sqeuclidean')
    wd_squared = ot.emd2(a, b, M)
    
    return np.sqrt(wd_squared)