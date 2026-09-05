from torch import nn
import torch.nn.functional as F


class QNetwork(nn.Module):

    def __init__(self, observation_dimensions, number_of_actions):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_dimensions, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, number_of_actions)
        )

    def forward(self, x):
        return self.model(x)
