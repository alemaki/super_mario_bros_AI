import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions: int, use_leaky_relu: bool = False, channel_multiplier: int = 1):
        super(ActorCritic, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32*channel_multiplier, kernel_size=8, stride=4),  # Change input_shape[0] to 1
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Conv2d(32*channel_multiplier, 64*channel_multiplier, kernel_size=4, stride=2),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Conv2d(64*channel_multiplier, 64*channel_multiplier, kernel_size=3, stride=1),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(input_shape), 512*channel_multiplier),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(512*channel_multiplier, n_actions),
        )
        self.critic = nn.Linear(512*channel_multiplier, 1)
        

    def _feature_size(self, shape):
        with torch.no_grad():
            return self.conv(torch.zeros(1, *shape)).view(1, -1).size(1)

    def forward(self, x):
        if x.isnan().any():
            print(x, "shape")
        x = self.conv(x)
        if x.isnan().any():
            print(x, "ater conv")
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if x.isnan().any():
            print(x, "ater fc")

        # Stabilize softmax input
        actor_output = self.actor[0](x)  # Get logits before softmax
        
        if actor_output.isnan().any():
            print(x, "actor output")
        actor_output = actor_output - torch.max(actor_output, dim=-1, keepdim=True).values  # Normalize
        probs = torch.nn.functional.softmax(actor_output / 2.0, dim=-1)
        if actor_output.isnan().any():
            print(x, "action probs")

        state_value = self.critic(x)
        if x.isnan().any():
            print(x, "state value")

        return probs, state_value