import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions: int, use_leaky_relu: bool = False, channel_multiplier: int = 1):
        super(ActorCritic, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32*channel_multiplier, kernel_size=8, stride=4),
            nn.LeakyReLU if use_leaky_relu else nn.ReLU,
            nn.Conv2d(32*channel_multiplier, 64*channel_multiplier, kernel_size=4, stride=2),
            nn.LeakyReLU if use_leaky_relu else nn.ReLU,
            nn.Conv2d(64*channel_multiplier, 64*channel_multiplier, kernel_size=3, stride=1),
            nn.LeakyReLU if use_leaky_relu else nn.ReLU
        )
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(input_shape), 512),
            nn.LeakyReLU if use_leaky_relu else nn.ReLU
        )

        self.actor = nn.Linear(512*channel_multiplier, n_actions)
        self.critic = nn.Linear(512*channel_multiplier, 1)

    def _feature_size(self, shape):
        with torch.no_grad():
            return self.conv(torch.zeros(1, *shape)).view(1, -1).size(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x.view(x.size(0), -1))
        return self.actor(x), self.critic(x)