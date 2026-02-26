import sys
import os
import torch
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_ddpm import NoisePredictor
from src.diffusion import DDPMScheduler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_kaggle_submission():
    NUM_SAMPLES = 5000 
    OUTPUT_FILE = "submission.csv"
    
    print(f"Generating {NUM_SAMPLES} samples")
    
    model = NoisePredictor(hidden_dim=512, num_layers=4).to(DEVICE)
    
    if os.path.exists("ddpm_best_ema.pth"):
        model.load_state_dict(torch.load("ddpm_best_ema.pth", map_location=DEVICE))
        print("Loaded BEST EMA model.")
    elif os.path.exists("ddpm_final_ema.pth"):
        model.load_state_dict(torch.load("ddpm_final_ema.pth", map_location=DEVICE))
        print("Loaded FINAL EMA model (Best not found).")
    else:
        print("Error: No model found.")
        return


    torch.manual_seed(2025)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2025)

    scheduler = DDPMScheduler(schedule='linear', device=DEVICE)
    
    print("Starting Full DDPM Sampling (1000 steps)")
    samples = scheduler.sample(model, shape=(NUM_SAMPLES, 2))
    samples = samples.cpu().numpy()
    
    df = pd.DataFrame(samples, columns=["x", "y"]) 
    df.insert(0, 'ID', range(1, NUM_SAMPLES + 1))
    
    if os.path.exists(OUTPUT_FILE):
        try:
            os.remove(OUTPUT_FILE)
        except:
            print(f"無法寫入{OUTPUT_FILE}")
            return

    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"成功: {OUTPUT_FILE}")
    print(f"欄位: {df.columns.tolist()}")

if __name__ == "__main__":
    generate_kaggle_submission()