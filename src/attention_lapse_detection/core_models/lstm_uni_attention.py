import torch
import torch.nn as nn

class LSTMUniAttention(nn.Module):
    """LSTM with uni-directional attention mechanism.

    Each timestep is scored as score_t = v^T tanh(W h_t + b).
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            num_classes: int,
            dropout: float
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.final_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor):
        outputs, _ = self.lstm(x)  # (batch, timesteps, hidden)
        scores = self.attention(outputs).squeeze(-1)  # (batch, timesteps)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (outputs * weights).sum(dim=1)  # (batch, hidden)
        return self.final_layer(self.dropout(context))