import torch.nn as nn
import torch

class FeedForward(nn.Module):

  def __init__(self, input_size, output_size,n_hidden_layers,depht):
    super().__init__()
    first_layer = nn.Linear(input_size,depht)
    first_layer.bias.data.fill_(0)
    l = [first_layer, nn.ReLU()]
    i = 0
    while i < n_hidden_layers:
      linear = nn.Linear(depht, depht)
      linear.bias.data.fill_(0)
      l.append(linear)
      l.append(nn.ReLU())
      i+=1
    last_layer = nn.Linear(depht, output_size)
    last_layer.bias.data.fill_(0)
    l.append(last_layer)

    self.model_list = nn.ModuleList(l)

  def forward(self,x):

    for layer in self.model_list:
      x = layer(x)
      
    return x.view(-1)