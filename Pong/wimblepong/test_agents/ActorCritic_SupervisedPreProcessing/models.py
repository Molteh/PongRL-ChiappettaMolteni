import torch
import torch.nn.functional as F
from torch.distributions import Categorical

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
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def flatten(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        return x

    def forward(self, x):

        # TODO: Actor
        x_ac = self.fc1_actor(x)
        x_ac = F.relu(x_ac)
        x_mean = self.fc2_mean(x_ac)
        x_probs = F.softmax(x_mean, dim=-1)
        dist = Categorical(x_probs)

        # TODO: Critic
        x_cr = self.fc1_critic(x)
        x_cr = F.relu(x_cr)
        value = self.fc2_value(x_cr)

        return dist, value
