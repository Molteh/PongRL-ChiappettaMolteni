import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple
import PIL.Image
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        # Layers
        self.conv1 = nn.Conv2d(
            in_channels=2,
            out_channels=16,
            kernel_size=8,
            stride=4,
            padding=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.fc1 = nn.Linear(
            in_features=3200,
            out_features=256,
        )
        self.fc2 = nn.Linear(
            in_features=256,
            out_features=3,
        )

        # Activation Functions
        self.relu = nn.ReLU()

    def flatten(self, x):
        batch_size = x.size()[0]
        x = x.view(batch_size, -1)
        return x

    def forward(self, x):
        #print(x.shape)
        # Forward pass
        x = self.relu(self.conv1(x))  # In: (80, 80, 2)  Out: (20, 20, 16)
        x = self.relu(self.conv2(x))  # In: (20, 20, 16) Out: (10, 10, 32)
        x = self.flatten(x)  # In: (10, 10, 32) Out: (3200,)
        x = self.relu(self.fc1(x))  # In: (3200,)      Out: (256,)
        x = self.fc2(x)  # In: (256,)       Out: (4,)

        return x


class Agent(object):
    def __init__(self, state_space, n_actions, replay_buffer_size=50000,
                 batch_size=32, hidden_size=12, gamma=0.98):
        self.n_actions = n_actions
        self.state_space_dim = state_space
        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayMemory(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.prev_obs = None
        self.train_device = "cpu"

    def update_network(self, updates=1):
        for _ in range(updates):
            self._do_network_update()

    def _do_network_update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = 1-torch.tensor(batch.done, dtype=torch.uint8)
        non_final_next_states = [s for nonfinal,s in zip(non_final_mask,
                                     batch.next_state) if nonfinal > 0]
        non_final_next_states = torch.cat(non_final_next_states)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask.bool()] = self.target_net(non_final_next_states).max(1)[0].detach()

        # Task 4: TODO: Compute the expected Q values
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(),
                                expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.from_numpy(next_state).float()
        state = torch.from_numpy(state).float()
        self.memory.push(state, action, next_state, reward, done)

    def load_model(self):
        weights = torch.load("model.mdl", map_location=torch.device("cpu"))
        self.policy_net.load_state_dict(weights, strict=False)  # ????

    def reset(self):
        self.prev_obs = None

    def preprocess(self, observation):
        """observation = observation[::2, ::2].mean(axis=-1)
        observation = np.expand_dims(observation, axis=-1)
        if self.prev_obs is None:
            self.prev_obs = observation
        stack_ob = self.prev_obs - observation
        #stack_ob = torch.from_numpy(stack_ob).float().unsqueeze(0)
        #stack_ob = stack_ob.transpose(1, 3)
        """

        if self.prev_obs is None:
            self.prev_obs = observation

        img_list = [observation, self.prev_obs]
        self.prev_obs = observation

        return self.phi_map(img_list)

    def get_action(self, observation, epsilon):

        x = observation

        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                x = torch.from_numpy(x).float()
                q_values = self.policy_net(x)
                return torch.argmax(q_values).item()
        else:
            return random.randrange(self.n_actions)

    def get_name(self):
        return "DQN agent"

    def phi_map(self, image_list, as_var=True):
        # Frame Skipping size
        k = len(image_list)

        im_tuple = tuple()
        for i in range(k):
            # Load single image as PIL and convert to Luminance
            pil_im = PIL.Image.fromarray(image_list[i]).convert('L')
            # Resize image
            pil_im = pil_im.resize((80, 80), PIL.Image.ANTIALIAS)
            # Transform to numpy array
            im = np.array(pil_im) / 255.
            pil_im.close()
            # Add processed image to tuple
            im_tuple += (im,)

        # Return tensor of processed images
        arr = self.tuple_to_numpy(im_tuple)

        # # Convert to Variable
        # if as_var:
        #     arr = Variable(torch.from_numpy(arr)).float()
        return arr

    def tuple_to_numpy(self, im_tuple):
        # Stack tuple of 2D images as 3D np array
        arr = np.dstack(im_tuple)
        # Move depth axis to first index: (height, width, depth) to (depth, height, width)
        arr = np.moveaxis(arr, -1, 0)
        # Make arr 4D by adding dimension at first index
        arr = np.expand_dims(arr, 0)
        return arr
