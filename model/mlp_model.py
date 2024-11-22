import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_features, hidden_features, output_features, dropout):
        super().__init__()

        self.sequential = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, output_features),
        )

    def forward(self, x):
        return self.sequential(x)
