import sys
import os
import torch
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import CheckerboardDataset
from src.model_ddpm import NoisePredictor
from src.diffusion import DDPMScheduler
from src.metrics import compute_energy_distance, compute_wasserstein_distance

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_ddim_experiments():
    print("Loading model and data")
    
    model = NoisePredictor(hidden_dim=512, num_layers=4).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load("ddpm_best_ema.pth"))
        print("Loaded BEST EMA model successfully.")
    except FileNotFoundError:
        print("Error: ddpm_best_ema.pth not found")
        try:
            model.load_state_dict(torch.load("ddpm_final_ema.pth"))
            print("Loaded FINAL EMA model successfully.")
        except:
            print("Error: No model found.")
            return

    scheduler = DDPMScheduler(device=DEVICE)
    
    eval_dataset = CheckerboardDataset(size=2000)
    real_data = torch.tensor(eval_dataset.data, dtype=torch.float32).to(DEVICE)
    
    results_steps = {'steps': [], 'ed': [], 'wd': []}
    results_etas = {'etas': [], 'ed': [], 'wd': []}

    test_steps = [1000, 500, 10, 1]
    print("\n Experiment 1: Varying Sampling Steps (eta=0)")
    
    for steps in test_steps:
        samples = scheduler.ddim_sample(model, shape=(2000, 2), ddim_timesteps=steps, eta=0.0)
        
        ed = compute_energy_distance(real_data, samples)
        wd = compute_wasserstein_distance(real_data, samples)
        
        print(f"Steps: {steps:4d} | ED: {ed:.4f} | WD: {wd:.4f}")
        results_steps['steps'].append(str(steps))
        results_steps['ed'].append(ed)
        results_steps['wd'].append(wd)

    test_etas = [0.0, 0.25, 0.5, 0.75, 1.0]
    fixed_step = 50 
    print(f"\nExperiment 2: Varying Eta (steps={fixed_step})")
    
    for eta in test_etas:
        samples = scheduler.ddim_sample(model, shape=(2000, 2), ddim_timesteps=fixed_step, eta=eta)
        
        ed = compute_energy_distance(real_data, samples)
        wd = compute_wasserstein_distance(real_data, samples)
        
        print(f"Eta: {eta:.2f} | ED: {ed:.4f} | WD: {wd:.4f}")
        results_etas['etas'].append(eta)
        results_etas['ed'].append(ed)
        results_etas['wd'].append(wd)

    print("\nPlotting results")
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results_steps['steps'], results_steps['ed'], marker='o', label='ED')
    plt.plot(results_steps['steps'], results_steps['wd'], marker='s', label='WD')
    plt.title("Effect of Sampling Steps (DDIM)")
    plt.xlabel("Denoising Steps")
    plt.ylabel("Distance (Lower is better)")
    plt.legend()
    plt.grid(True)
    plt.gca().invert_xaxis()
    
    plt.subplot(1, 2, 2)
    plt.plot(results_etas['etas'], results_etas['ed'], marker='o', label='ED')
    plt.plot(results_etas['etas'], results_etas['wd'], marker='s', label='WD')
    plt.title(f"Effect of Eta (Steps={fixed_step})")
    plt.xlabel("Eta (Stochasticity)")
    plt.ylabel("Distance")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("ddim_experiment_results.png")
    print("Results saved to ddim_experiment_results.png")

if __name__ == "__main__":
    run_ddim_experiments()