import torch
import torch.nn as nn

class MazeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):

        super(MazeMLP, self).__init__()

        # input encoding -> hidden layer -> 5 output (action space)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 5)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        logits = self.fc2(out)
        return logits
