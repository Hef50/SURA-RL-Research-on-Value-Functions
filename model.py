import torch
import torch.nn as nn

class MazeMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3):
        # initializes nn.Module superclass
        super(MazeMLP, self).__init__()

        # input encoding -> hidden layers -> 5 output (action space)
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 5))
        self.network = nn.Sequential(*layers) # splat unpack the layer list into comma separated arguments 
        
    
    def forward(self, x):
        return self.network(x)

class MazeCNN(nn.Module):
    def __init__(self, d=7, hidden_dim=128):
        super(MazeCNN, self).__init__()

        # NO POOLING HERE bc 8x8 too small, pooling mostly for images and classification for position invariance
        # but we don't want position invariance here bc we need strict location tracking

        # layer 1 - processes the 3 raw channels into 32 spatial feature maps
        # using padding=1 preserves the width and height of the grid boundary
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        # layer 2 - deepens feature extraction from 32 maps to 64 maps
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        # calculate the total flattened size after convolutions
        # since padding=1 preserves dimensions, spatial size remains D x D
        self.flattened_dim = 64 * d * d

        # dense projection layers to compute action scores
        self.fc1 = nn.Linear(self.flattened_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))

        # flattening, out.view changes dimensions of tensor
        # out.size(0) locks the batch dimension and keeps data separated in batches properly
        # -1 flag tells it to figure out math automatically for rest of tensors
        out = out.view(out.size(0), -1)

        # MLP classification
        out = self.relu3(self.fc1(out))
        logits = self.fc2(out)
        return logits