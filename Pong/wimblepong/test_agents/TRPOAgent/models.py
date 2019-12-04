import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class DQNSoftmax(nn.Module):
  def __init__(self, output_size):
    super(DQNSoftmax, self).__init__()

    self.conv1 = nn.Conv2d(2, 16, kernel_size=8, stride=4)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
    self.fc = nn.Linear(8096, 256)
    self.head = nn.Linear(256, output_size)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    out = F.relu((self.conv1(x)))
    out = F.relu(self.conv2(out))
    out = F.relu(self.fc(out.view(out.size(0), -1)))
    out = self.softmax(self.head(out))
    return out


class DQNRegressor(nn.Module):
  def __init__(self):
    super(DQNRegressor, self).__init__()

    self.conv1 = nn.Conv2d(2, 16, kernel_size=8, stride=4)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
    self.fc = nn.Linear(8096, 256)
    self.head = nn.Linear(256, 1)

  def forward(self, x):
    out = F.relu((self.conv1(x)))
    out = F.relu(self.conv2(out))
    out = F.relu(self.fc(out.view(out.size(0), -1)))
    out = self.head(out)
    return out

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
