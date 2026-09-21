import torch
import torch.nn as nn


class LSTM(nn.Module):
    "LSTM with pooled by the final hidden state"

    def __init__(
            self,
            input_size:int,
            hidden_size:int,
            num_layers:int,
            num_classes:int,
            dropout: float
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size= input_size,
            hidden_size= hidden_size,
            num_layers= num_layers,
            dropout= dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.final_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        _, (hidden, _) = self.lstm(x) # (num_layers, batch, hidden)
        return self.final_layer(self.dropout(hidden[-1]))