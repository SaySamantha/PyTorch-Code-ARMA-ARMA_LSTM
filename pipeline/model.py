# This file contains model-related classes and functions
import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, dropout, norm, num_layers):
        super(LSTMModel, self).__init__()
        self.temporal_module = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.norm = norm(hidden_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.prediction_module = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input_features):
        hidden_features, _ = self.temporal_module(input_features)
        hidden_features = hidden_features[:, -1, :]     # Get the hidden state at the final timestep
        hidden_features = self.norm(hidden_features)
        hidden_features = self.activation(hidden_features)
        hidden_features = self.dropout(hidden_features)
        output_features = self.prediction_module(hidden_features)
        
        return output_features
