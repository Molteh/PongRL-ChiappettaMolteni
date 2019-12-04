from Pong_NN import PongNN as PNN
import numpy as np
import random
import torch


class NNAgent(object):
    def __init__(self):
        self.name = "NNAgent"
        self.NNX = PNN()
        self.NNY = PNN()
        self.NNmyY = PNN()

    def get_name(self):
        """
        Interface function to retrieve the agents name
        """
        return self.name

    def load_model(self):
        # import X weights
        weights = torch.load("weights_XNN.mdl", map_location=torch.device("cpu"))
        self.NNX.load_state_dict(weights, strict=False)

        # import Y weights
        weights = torch.load("weights_YNN.mdl", map_location=torch.device("cpu"))
        self.NNY.load_state_dict(weights, strict=False)

        # import myY weights
        weights = torch.load("weights_myYNN.mdl", map_location=torch.device("cpu"))
        self.NNmyY.load_state_dict(weights, strict=False)

    # pre process 200x200x3 uint8 frame into 100000 (100x100) 1D float vector
    def preprocess(self, frame):
        frame = frame[::2, ::2, 0]  # down sample by factor of 2
        frame[frame == 43] = 0  # erase background (background type 1)

        frame[frame != 0] = 1  # everything else (paddles, ball) just set to 1
        return frame.astype(np.float).ravel()

    def get_action(self, observation):
        """
        Interface function that returns the action that the agent took based
        on the observation ob
        """

        observation = self.preprocess(observation)

        observation = torch.tensor(observation)

        # Wrap them in a Variable object
        #observation = Variable(observation)

        observation = observation.float()

        # get myY
        my_y = self.NNmyY(observation)

        # get Y
        y = self.NNY(observation)

        # get noisy x
        x = self.NNX(observation)

        my_y = my_y.detach().numpy()
        x = x.detach().numpy()
        y = y.detach().numpy()

        # Get the ball position in the game arena
        ball_y = y + (random.random()*np.log(x)-np.log(x/2))

        # Compute the difference in position and try to minimize it
        y_diff = my_y - ball_y
        if abs(y_diff) < 2:
            action = 0  # Stay
        else:
            if y_diff > 0:
                action = 1  # Up
            else:
                action = 2  # Down

        return action

    def reset(self):
        # Nothing to done for now...
        return
