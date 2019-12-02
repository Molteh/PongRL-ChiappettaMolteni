import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.fc2_value = torch.nn.Linear(self.hidden, 1)
        self.sigma = torch.nn.Parameter(torch.tensor([10.]))  # Learn variance as a parameter of the network
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, variance):

        # TODO: Actor
        x = self.fc1_actor(x)
        x = F.relu(x)
        x_mean = self.fc2_mean(x)
        dist = Normal(x_mean, torch.sqrt(variance)) # TODO: Normal dist
        # x_probs = F.softmax(x_mean, dim=-1)  # TODO: Soft-max distibution
        # dist = Categorical(x_probs)

        # TODO: Critic
        x_cr = self.fc1_critic(x)
        x_cr = F.relu(x_cr)
        value = self.fc2_value(x_cr)

        return dist, value
