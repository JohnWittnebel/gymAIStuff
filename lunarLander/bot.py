import numpy as np
from random import uniform
import math
from torch import nn
from torch.masked import masked_tensor
import torch

hiddenNodes = 10

class NeuralNetwork(nn.Module):
    def __init__(self, inodes, onodes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inodes, hiddenNodes),
            nn.ReLU(),
            nn.Linear(hiddenNodes, hiddenNodes),
            nn.ReLU(),
            nn.Linear(hiddenNodes, onodes)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

