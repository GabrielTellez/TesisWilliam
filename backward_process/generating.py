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

    model, loss_hist_train, val_hist_train, scaler = TrainModel(timesteps, ndata, initial_distribution) 

    PlotTrainval(ndata, loss_hist_train, val_hist_train)

    print("Backward process started...")
 
    for t in range(1,timesteps):

        times = np.array(timesteps - t).repeat(ndata)
        
        feat = np.vstack((distros[t-1], times)).T
        scaled_feat = scaler.transform(feat)

        features = torch.tensor(scaled_feat, dtype=torch.float32)

        guessed_noise = model(features).detach().numpy().flatten()

        beta_hat = beta[t] * (1 - np.prod(alpha[:t-1]))/(1 - np.prod(alpha[:t]))
        noise =  np.sqrt(beta_hat) * np.random.normal(0,1,ndata)
        # el tiempo está al revez en alpha y beta, debería ser timesteps-t
        distros[t] = 1/np.sqrt(alpha[t]) * (distros[t-1] - guessed_noise* beta[t]/(np.sqrt(1 - np.prod(alpha[:t]))) ) + noise

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

