[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UPVOpvc0)
# ML Assignment 4: Generative Models

This repository contains the implementation for Assignment 4 of the Machine Learning (CSIE5043) course, Fall 2025, at National Taiwan University. In this assignment, we study generative modeling on 2D synthetic datasets by implementing and comparing three representative model families: GAN, DDPM, and DDIM.

Student ID: b10801011

Name:張鈞傑

Kaggle Score: -0.0013 

## Setup
To install all required dependencies, simply run:
```
uv sync
```


## Training
To start training the models, you can execute the scripts in the scripts/ directory:
bash scripts/run_problem_1.sh   #GAN
bash scripts/run_problem_2a.sh  #DDPM

## Generating Submission
To generate predictions for submission and run inference experiments:
bash scripts/run_problem_2b.sh  # For DDIM Inference & Plotting

## Experiment Results
Main Results
<table> <tr> <td>Methods</td> <td>#Step</td> <td>ED</td> <td>WD</td> </tr> <tr> <td>GAN</td> <td>1</td> <td>0.03</td> <td>0.35</td> </tr> <tr> <td>DDPM</td> <td>1000</td> <td><strong>0.00049</strong></td> <td>0.28 (Kaggle: -0.0013)</td> </tr> <tr> <td>DDIM</td> <td>1000</td> <td>0.0005</td> <td>0.28</td> </tr> <tr> <td>DDIM</td> <td>500</td> <td>0.0005</td> <td>0.21</td> </tr> <tr> <td>DDIM</td> <td>10</td> <td>0.04</td> <td>0.70</td> </tr> <tr> <td>DDIM</td> <td>1</td> <td>1.05</td> <td>2.85</td> </tr> </table>

Observation / Insight:
Best Performance: The DDPM (1000 steps) achieved the best local Energy Distance of 0.00049, demonstrating high-fidelity generation.
Step Trade-off: As shown in the DDIM experiments, decreasing the sampling steps from 1000 to 1 leads to a significant increase in approximation error (higher Energy Distance). The 1000-step DDIM matches the baseline DDPM quality.
GAN vs. Diffusion: While GAN is fast (1 step), the converged Diffusion model (DDPM) provided more stable and metric-superior results for this specific distribution.

### Ablation Study
####DDPM
#Step: 1000
<table> <tr> <td>Time Embedding</td> <td>ED</td> <td>WD</td> </tr> <tr> <td>Sinusoidal Positional Embedding</td> <td><strong>0.00049</strong></td> <td>0.28</td> </tr> </table>
Observation / Insight:
We adopted Sinusoidal Position Embeddings for the final model. This strategy, combined with the Linear Schedule, allowed the NoisePredictor to effectively capture timestep information.
Architecture Note: Replacing Dropout with LayerNorm in Residual Blocks significantly stabilized gradients and sharpened boundaries.

#### DDIM
#Step: 50

<table> <tr> <td>eta</td> <td>ED</td> <td>WD</td> </tr> <tr> <td>0</td> <td>0.003</td> <td>0.33</td> </tr> <tr> <td>0.25</td> <td><strong>0.002</strong></td> <td><strong>0.25</strong></td> </tr> <tr> <td>0.5</td> <td>0.002</td> <td>0.28</td> </tr> <tr> <td>0.75</td> <td>0.004</td> <td>0.29</td> </tr> <tr> <td>1</td> <td>0.006</td> <td>0.38</td> </tr> </table>
Observation / Insight:
Stochasticity Impact: As visualized in the results , lower Eta values (closer to deterministic) yielded better metrics for this limited-step scenario (50 steps).
Diversity vs. Precision: Eta = 1 (Full DDPM stochasticity) introduces more noise during sampling, resulting in the highest Wasserstein Distance (0.38) in this experiment. The path with slight stochasticity (Eta=0.25) achieved the best balance (WD=0.25).

### Learning Curve Analysis
#### Training steps vs Training Loss
**GAN (Generator & Discriminator):**
<br>
![Loss](images/loss_g.png) ![Loss](images/loss_d.png)

**DDPM:**
<br>
![DDPM Loss](images/ddpm_loss.png)

Observation / Insight:
GAN: The GAN loss curves exhibit significant oscillation (the "min-max" game), which is typical for adversarial training.
DDPM: In contrast, the DDPM loss curve shows a smooth, consistent decrease, indicating that the model successfully learned to minimize the noise prediction error (MSE) over time.

#### Per 100 epochs vs Energy Distance
**GAN:**
<br>
![Metrics](images/Energy Distance.png)

**DDPM:**
<br>
![DDPM Metrics](images/ddpm_ED.png)

Observation / Insight:
Convergence: The DDPM Energy Distance (ED) successfully dropped below 0.001, confirming the model generates samples that statistically match the ground truth distribution.
Stability: The DDPM curve is smoother compared to the GAN's initial training phase.

#### Per 100 epochs vs 2-Wasserstein Distance
**GAN:**
<br>
![Metrics](images/W Distance.png)

**DDPM:**
<br>
![DDPM Metrics](images/ddpm_WD.png)

Observation / Insight:
Similar to the Energy Distance, the Wasserstein Distance for DDPM shows a steady decline, validating the effectiveness of the Linear Schedule (beta 1e-4 to 0.02) and LayerNorm modifications.

### Visualization


#### GIF of training progression (learned distribution over epochs)
**DDPM Reverse Process:**
<br>
![Training Process](images/training.gif)

Observation / Insight:
The GIF demonstrates the reverse diffusion process, denoising from pure Gaussian noise to the target checkerboard distribution. The transition is coherent, showing the model effectively recovering the structure.

#### Figure of final learned distribution
**GAN (Epoch 2000):**
<br>
![GAN_train](images/gan_epoch_2000.png)

**DDPM (Epoch 3000 - Best Model):**
<br>
![Training Pic](images/epoch_3000.png)

**DDIM (Effect of Steps & Eta):**
<br>
![DDIM Steps](ddim_experiment_results.png)

Observation / Insight:
Fidelity: The DDPM (Epoch 3000) image shows very sharp square boundaries and even density, outperforming the GAN which shows slightly fuzzier points.
DDIM Analysis: The right-hand figure confirms that fewer steps (e.g., 1 or 10) significantly degrade the structure (higher ED), while Eta=0 is optimal for metric performance in this setup.
