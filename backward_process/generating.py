import numpy as np
import torch
import matplotlib.pyplot as plt

from forward_process.generate_noised_data import BetaSchedule
from neural_network.training_nn import TrainModel

from save_plot.save_files import SaveCSV

def Generate(initial_distribution, timesteps, ndata):

    """
    Emulates the backward process of a diffusion model to generate data. The 
    dynamics of the data for each timestep is stored in an array called distros
    (shape=(timesteps, ndata)). The first row is filled with random noise.
    """
    
    beta = BetaSchedule(timesteps)
    alpha = 1 - beta

    distros = np.zeros((timesteps,ndata))
    distros[0] = np.random.normal(0, 1, ndata)

    train_steps = 1000

    model, loss_hist_train, val_hist_train, scaler = TrainModel(train_steps, ndata, initial_distribution) 

    PlotTrainval(ndata, loss_hist_train, val_hist_train)

    print("Backward process started...")
 
    for t in range(1,timesteps):
        
        s = timesteps - t

        times = np.array(s).repeat(ndata)
        
        feat = np.vstack((distros[t-1], times)).T
        scaled_feat = scaler.transform(feat)

        features = torch.tensor(scaled_feat, dtype=torch.float32)

        guessed_noise = model(features).detach().numpy().flatten()

        beta_hat = beta[s] * (1 - np.prod(alpha[:s-1]))/(1 - np.prod(alpha[:s]))
        noise =  np.sqrt(beta_hat) * np.random.normal(0,1,ndata)
        
        distros[t] = 1/np.sqrt(alpha[s]) * (distros[t-1] - guessed_noise* beta[s]/(np.sqrt(1 - np.prod(alpha[:s]))) ) + noise

    SaveCSV(distros, "generated_data")

def PlotTrainval(ndata, loss_hist_train, loss_hist_valid):
    """
    Plots learning curves of training and validation sets
    """

    fig, ax = plt.subplots()

    ax.set_xlabel("epochs")
    ax.set_ylabel("MSE")
    ax.set_ylim(0, max(loss_hist_train.max(), loss_hist_valid.max())) 
    ax.set_title("Training and validation errors")

    ax.plot(loss_hist_train,label='train')
    ax.plot(loss_hist_valid,label='valid')
    ax.legend(fontsize='large')

    fig.savefig('figures/losses/'+str(ndata)+'.svg')

    print(f"Training and validation losses plotted and saved to figures/losses/{ndata}.svg")

