import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

torch.cuda.is_available()

device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'

torch.set_default_device(device)

from neural_network.neural_network import FeedForward
#from neural_network.neural_network import ConvNet
from forward_process.generate_noised_data import GenerateNoisedData
from neural_network.preprocessing import Preprocessing
from save_plot.save_files import SaveCSV

def Train(learning_rate, model, num_epochs, train_dl, valid_dl):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    loss_hist_train = np.zeros(num_epochs)
    loss_hist_valid = np.zeros(num_epochs)

    for epoch in range(num_epochs):

      model.train()
# Error: no mezclar train y validation al tiempo. Primero se entrena sobre todo
# el dataloader y luego se valida.
      for (x_train_batch, y_train_batch), (x_val_batch, y_val_batch) in zip(train_dl, valid_dl):

        train_pred = model(x_train_batch).view(-1)
            #Define loss function
        train_loss = loss_fn(train_pred, y_train_batch)
            #Backpropagation
        train_loss.backward()
            #Apply gradient to the weights
        optimizer.step()
            #Make gradients zero. Error: debe ir al inicio del bucle
        optimizer.zero_grad()

        val_pred = model(x_val_batch).view(-1)
        val_loss = loss_fn(val_pred, y_val_batch)
# error: no está promediando sobre todo el batch la loss y solo reporta la del
# último batch
        loss_hist_train[epoch] = train_loss.item()
        loss_hist_valid[epoch] = val_loss.item()

    return loss_hist_train, loss_hist_valid

def TrainModel(timesteps, ndata, initial_distribution):

    model = FeedForward(input_size=2,output_size=1,n_hidden_layers=2,depht=200).to(device)

    features, noise = GenerateNoisedData(timesteps, ndata, initial_distribution)

    features = features.reshape(-1,2)
    scaler = StandardScaler()

    features = scaler.fit_transform(features)
# noise hay tambien que reshape
    train_dl, valid_dl, test_feature, test_target = Preprocessing(features, noise)



    loss_hist_train, loss_hist_valid = Train(learning_rate=0.01, model=model, num_epochs=30,
                                           train_dl=train_dl, valid_dl=valid_dl
                                           )

    pred = model(test_feature).view(-1)

    loss_fn = nn.MSELoss()

    test_loss = loss_fn(pred, test_target).item()

    print(f'test error:  {test_loss}')

    return model, loss_hist_train, loss_hist_valid, scaler

#########################


from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> float:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    training loss
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss = 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Optimizer zero grad
        optimizer.zero_grad()

        # 2. Forward pass
        y_pred = model(X)

        # 3. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Adjust metrics to get average loss per batch
    train_loss = train_loss / len(dataloader)
    return train_loss

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    testing loss 
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss = 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()

    # Adjust metrics to get average loss per batch 
    test_loss = test_loss / len(dataloader)
    return test_loss

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              test_loss: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              test_loss: [1.2641, 1.5706]}
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "test_loss": [],
    }
    
    # Make sure model on target device
    model.to(device)

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                 dataloader=train_dataloader,
                                 loss_fn=loss_fn,
                                 optimizer=optimizer,
                                 device=device)
        test_loss = test_step(model=model,
                                dataloader=test_dataloader,
                                loss_fn=loss_fn,
                                device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4e} | "
          f"test_loss: {test_loss:.4e} | "
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)

    # Return the filled results at the end of the epochs
    return results
