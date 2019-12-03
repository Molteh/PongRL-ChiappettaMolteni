from torch.autograd import Variable
import torch
import torch.optim as optim
import time
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from Pong_NN import PongNN
import PongDataset
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import argparse
import wimblepong
import random


def createLossAndOptimizer(net, learning_rate=0.001):
    # Loss function
    loss = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return loss, optimizer


# DataLoader takes in a dataset and a sampler for loading (num_workers deals with system level memory)
def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    return train_loader


def preprocess(frame):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    frame = frame[::2, ::2, 0]  # downsample by factor of 2
    frame[frame == 43] = 0  # erase background (background type 1)
    frame[frame != 0] = 1  # everything else (paddles, ball) just set to 1
    return frame.astype(np.float).ravel()


def trainNet(net, batch_size, n_epochs, learning_rate):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    # Get training data
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)

    # Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)

    # Time for printing
    training_start_time = time.time()

    # Loop for n_epochs
    for epoch in range(n_epochs):

        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):

            # Get inputs
            inputs, labels = data

            # Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)

            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss_size.data[0]
            total_train_loss += loss_size.data[0]

            # Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))
                # Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in val_loader:
            # Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)

            # Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data[0]

        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

############################################################

"""
This is an example on how to use the two player Wimblepong environment with one
agent and the SimpleAI
"""

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--housekeeping", action="store_true", help="Plot, player and ball positions and velocities at the end of each episode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps

# Number of episodes/games to play
episodes = 1000

# Define the player
player_id = 1
# Set up the player here. We used the SimpleAI that does not take actions for now
player = wimblepong.SimpleAi(env, player_id)

# Housekeeping
samples = []
win1 = 0

for i in range(0,episodes):
    done = False
    while not done:
        # action1 is zero because in this example no agent is playing as player 0
        action1 = player.get_action()
        ob1, rew1, done, info = env.step(action1)
        if args.housekeeping and np.random.uniform() > 0.1:
            samples.append([preprocess(ob1), env.ball.x, env.ball.y])
        # Count the wins
        if rew1 == 10:
            win1 += 1
        if not args.headless:
            env.render()
        if done:
            observation = env.reset()
            plt.close()  # Hides game window

            print("episode {} over. Broken WR: {:.3f}".format(i, win1/(i+1)))
            if i % 5 == 4:
                env.switch_sides()
                print("Current samples: ", len(samples))
    if len(samples) >= 2500:
        break
print("Sampling done")

data = samples
random.shuffle(data)

train_set = PongDataset.PongDataset(data[:2000])
val_set = PongDataset.PongDataset(data[2000:2500])

# Training
n_training_samples = 2000
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

# Validation
n_val_samples = 500
val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, sampler=val_sampler, num_workers=2)

CNN = PongNN()
trainNet(CNN, batch_size=32, n_epochs=5, learning_rate=0.001)
