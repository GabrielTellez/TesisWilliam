"""
Contains various utilities for PyTorch model training.
"""
import torch
from pathlib import Path
from neural_network.neural_network import FeedForward

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
  
  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  
  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
            
def load_model(filename: str,
              input_size=2,
              output_size=1,
              n_hidden_layers=2,
              depth=200
              ) -> torch.nn.Module:
  """Loads a classification model.
  
  Args:
    filename (str): filename of the saved model.
    input_size (int): input size of the model. Default 2.
    output_size (int): output size of the model. Default 1.
    n_hidden_layers (int): number of hidden layers. Default 2.
    depth (int): depth of each hidden layer. Default 200.

  Returns:
    model (torch.nn.Module): the loaded model. 
  """

  model = FeedForward(input_size=input_size,
                      output_size=output_size,
                      n_hidden_layers=n_hidden_layers,
                      depht=depth)
  model.load_state_dict(torch.load(f=filename))

  return model

  