import torch
from torch import nn
import torch.nn.functional as F


class DynamicLayer(nn.Module):
    def __init__(self, input_dim, output_dim, max_depth):
        super(DynamicLayer, self).__init__()
        self.max_depth = max_depth
        self.layers = nn.ModuleList([nn.Linear(input_dim if i == 0 else output_dim, output_dim) for i in range(max_depth)])
        self.gates = nn.ModuleList([nn.Linear(output_dim, 1) for _ in range(max_depth)])

    def forward(self, x, depth=0):
        if depth >= self.max_depth:
            return x
        x = F.relu(self.layers[depth](x))
        gate_status = torch.sigmoid(self.gates[depth](x)).mean()
        if gate_status > 0.5:
            return self.forward(x, depth + 1)
        else:
            return x


