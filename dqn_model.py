import torch
import torch.nn as nn
import torch.optim as optim

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.LAYERS = 128
        self.fc1 = nn.Linear(input_dim, self.LAYERS)
        self.fc2 = nn.Linear(self.LAYERS, self.LAYERS)
        self.fc3 = nn.Linear(self.LAYERS, self.LAYERS)
        self.fc4 = nn.Linear(self.LAYERS, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

