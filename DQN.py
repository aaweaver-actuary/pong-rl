import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self
                 , input_dim: int
                 , output_dim: int
                 , layer_nodes: list = [256, 128]
                 , dropout: float = 0.3
                 , leaky_relu: float = 0.2
                 , optimizer: str = 'adam'):
      super(DQN, self).__init__()
      self.input_dim = input_dim
      self.output_dim = output_dim
      self.layers = nn.ModuleList()

      # input layer (add dropout and leaky relu)
      self.layers.append(nn.Linear(self.input_dim, layer_nodes[0]))
      self.layers.append(nn.LeakyReLU(leaky_relu))
      self.layers.append(nn.Dropout(dropout))

      # hidden layers 
      for i, layer in enumerate(layer_nodes[1:]):
          self.layers.append(nn.Linear(layer_nodes[i], layer))
          self.layers.append(nn.LeakyReLU(leaky_relu))
          self.layers.append(nn.Dropout(dropout))
    
      # output layer
      self.layers.append(nn.Linear(layer_nodes[-1], self.output_dim))

      # optimizer
      if optimizer.lower() == 'adam':
          self.optimizer = torch.optim.Adam(self.parameters())
      elif optimizer.lower() == 'sgd':
          self.optimizer = torch.optim.SGD(self.parameters())
      elif optimizer.lower() == 'rmsprop':
          self.optimizer = torch.optim.RMSprop(self.parameters())
      elif optimizer.lower() == 'adagrad':
          self.optimizer = torch.optim.Adagrad(self.parameters())
      else:
          raise ValueError('Optimizer not recognized')
      
      # loss function
      self.loss_fn = nn.MSELoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x