import random
import torch
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('trajectory'))


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
        """
        Sample a batch of trajectories from the buffer. If they are of unequal length
        (which is likely), the trajectories will be padded with zero-reward transitions.
        Parameters
        ----------
        batch_size : int
            The batch size of the sample.
        Returns
        -------
        list of Transition's
            A batched sampled trajectory.
        """
        batched_trajectory = ([], [], [])
        trajectories = random.sample(self.memory, batch_size)
        for trajectory in trajectories:
            batched_trajectory[0].append(trajectory[0])
            batched_trajectory[1].append(trajectory[0])
            batched_trajectory[2].append(trajectory[0])
        return batched_trajectory

    @staticmethod
    def extend(transition):
        """
        Generate a new zero-reward transition to extend a trajectory.
        Parameters
        ----------
        transition : Transition
            A terminal transition which will become the new transition's previous
            transition in the trajectory.
        Returns
        -------
        Transition
            The new transition that can be used to extend a trajectory.
        """
        if not transition.done[0, 0]:
            raise ValueError("Can only extend a terminal transition.")
        exploration_statistics = torch.ones(transition.exploration_statistics.size()) \
                                 / transition.exploration_statistics.size(-1)
        transition = Transition(states=transition.next_states,
                                actions=transition.actions,
                                rewards=torch.FloatTensor([[0.]]),
                                next_states=transition.next_states,
                                done=transition.done,
                                exploration_statistics=exploration_statistics)
        return transition