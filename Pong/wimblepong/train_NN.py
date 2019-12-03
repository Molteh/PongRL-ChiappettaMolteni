from torch.autograd import Variable
import torch
import torch.optim as optim
import time
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from Pong_NN import PongNN
import PongDataset
import matplotlib.pyplot as plt
import gym
import argparse
import wimblepong
import random


def create_loss_and_optimizer(net, lr=0.001):

    # Optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # Loss function
    loss = torch.nn.MSELoss()  # this is for regression mean squared loss

    return loss, optimizer


# dataLoader takes in a data set and a sampler for loading
def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    return train_loader


# pre process 200x200x3 uint8 frame into 100000 (100x100) 1D float vector
def preprocess(frame):
    frame = frame[::2, ::2, 0]  # down sample by factor of 2
    frame[frame == 43] = 0  # erase background (background type 1)

    frame[frame != 0] = 1  # everything else (paddles, ball) just set to 1
    return frame.astype(np.float).ravel()


def train_net(net, batch_size, n_epochs, lr):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", lr)
    print("=" * 30)

    # Get training data
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)

    # Create our loss and optimizer functions
    loss, optimizer = create_loss_and_optimizer(net, lr)

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
            inputs = Variable(inputs)
            labels = Variable(labels)
            labels = labels.unsqueeze(-1)

            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = net(inputs.float())
            loss_size = loss(outputs, labels.float())
            loss_size.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss_size.data
            total_train_loss += loss_size.data

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
            labels = labels.unsqueeze(-1)

            # Forward pass
            val_outputs = net(inputs.float())
            val_loss_size = loss(val_outputs, labels.float())
            total_val_loss += val_loss_size.data

        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


def complete_sample():
    samples = []
    for x in range(100):
        for y in range(100):
            sample = np.zeros((100, 100), dtype=int)
            sample[x, y] = 1
            sample = sample.flatten()
            samples.append([sample, x, y])
    random.shuffle(samples)
    return samples


def sample(env, batch_size, prob):
    # Define the player
    player_id = 1
    # Set up the player here. We used the SimpleAI that does not take actions for now
    player = wimblepong.SimpleAi(env, player_id)

    samples = []
    i = 0

    # run until the data set has been sampled
    while True:
        done = False
        while not done:
            action1 = player.get_action()
            ob1, rew1, done, info = env.step(action1)
            if args.housekeeping and np.random.uniform() > prob:
                samples.append([preprocess(ob1), env.ball.x, env.player1.y])
            if not args.headless:
                env.render()
            if done:
                plt.close()  # Hides game window

                print("episode {} over.".format(i))
                if i % 5 == 4:
                    # env.switch_sides() do not switch sides
                    print("Current samples: ", len(samples))

                env.reset()
        if len(samples) >= batch_size:
            break
        i += 1
    print("Sampling done")

    random.shuffle(samples)
    return samples[:batch_size]

############################################################


# command line args
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--housekeeping", action="store_true", help="Save samples")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps

# PARAMETERS
train = True
load_model = False

train_set_size = 25000
val_set_size = 5000
test_set_size = 1000

train_sampling_prob = 0.8
test_sampling_prob = 0.9

n_epochs = 250
learning_rate = 0.001
val_batch_size = 128
train_batch_size = 32

# END PARAMETERS

if train:

    # generate training-validation set
    data = sample(env, train_set_size+val_set_size, train_sampling_prob)

    train_set = PongDataset.PongDataset(data[:train_set_size])
    val_set = PongDataset.PongDataset(data[train_set_size:train_set_size + val_set_size])

    # Training
    n_training_samples = train_set_size
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

    # Validation
    n_val_samples = val_set_size
    val_sampler = SubsetRandomSampler(np.arange(n_val_samples, dtype=np.int64))

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, sampler=val_sampler, num_workers=2)

# initialize the network
CNN = PongNN()

# load previous model
if load_model:
    weights = torch.load("weights_myYNN.mdl", map_location=torch.device("cpu"))
    CNN.load_state_dict(weights, strict=False)

if train:
    train_net(CNN, batch_size=train_batch_size, n_epochs=n_epochs, lr=learning_rate)
    torch.save(CNN.state_dict(), "weights_%s.mdl" % "myYNN")

# always run test at the end
test_set = sample(env, test_set_size, test_sampling_prob)

tot_error = 0
for sample in test_set:
    x = Variable(torch.tensor(sample[0]))
    out = CNN(x.float()).data.numpy().flatten()[0]
    error = abs(out - sample[2])
    tot_error += error
    print(out, "-", sample[2], "Error: ", error)
print("Avg error: ", tot_error/len(test_set))
