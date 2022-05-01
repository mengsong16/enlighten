import numpy as np
import torch
import torch.nn as nn
from gym import spaces

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, hidden_layer=2, dropout=0):
        super(MLPEncoder, self).__init__()
        
        # mlp module
        assert hidden_layer >= 1, "Error: Must have at least one hidden layers"
        # hidden layer 1 (input --> hidden): linear+relu+dropout
        self.mlp_module = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()]

        if dropout > 0:    
            self.mlp_module.append(nn.Dropout(dropout))
        
        # hidden layer 2 to n (hidden --> hidden): linear+relu+dropout
        for _ in range(hidden_layer-1):
            self.mlp_module.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ])

            if dropout > 0:    
                self.mlp_module.append(nn.Dropout(dropout))
        
        # last layer n+1 (hidden --> output)
        self.mlp_module.extend([
            nn.Linear(hidden_dim, output_dim)
        ])

        self.mlp_module = nn.Sequential(*self.mlp_module)
    
    def get_device(self):
        return next(self.parameters()).device

    def from_numpy_to_tensor(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        
        x = x.to(self.get_device())
        return x

    def forward(self, states):
        # ensure states are tensors on the same device
        states = self.from_numpy_to_tensor(states)

        return self.mlp_module(states)
    
    # for evaluation, return numpy arrays
    def evaluate_state(self, states):
        with torch.no_grad():
            output = self.forward(states)

            return output.cpu().numpy()
