import wimblepong
import gym
import numpy as np
from matplotlib import pyplot as plt
from itertools import count
import torch
import logging
import sys
import os

from test_agents.DQNAgent.agent_train import Agent as DQNAgent

# set up logging
logging.basicConfig(level=logging.INFO, filename='dqn.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.info("log file for DQN agent training")

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")

TARGET_UPDATE = 20
glie_a = 5000
num_episodes = 15000
hidden = 64
gamma = 0.95
replay_buffer_size = 50000
batch_size = 128

###
wins = 0

###
resume = True # resume from previous checkpoint?

# Get number of actions from gym action space
n_actions = env.action_space.n
state_space_dim = env.observation_space.shape


# Task 4 - DQN
agent = DQNAgent(state_space_dim, n_actions, replay_buffer_size, batch_size,
               hidden, gamma)

if resume:
    dir = "/Users/molte/Desktop/Aalto/Reinforcement learning/Project/PongRL-ChiappettaMolteni/Pong/wimblepong/test_agents/DQNAgent"
    sys.path.insert(0, dir)
    orig_wd = os.getcwd()
    os.chdir(dir)
    agent.load_model()
    print("Resuming from previous model")
    logging.info("Resuming from previous model")
    os.chdir(orig_wd)
    del sys.path[0]


# Training loop
cumulative_rewards = []
for ep in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()

    # PRE PROCESS THE STATE
    state = agent.preprocess(state)

    done = False
    eps = glie_a/(glie_a+ep)
    cum_reward = 0

    i = 0

    while not done:
        # Select and perform an action
        action = agent.get_action(state, eps)
        next_state, reward, done, _ = env.step(action)
        cum_reward += reward

        next_state = agent.preprocess(next_state)

        # Task 4: Update the DQN
        agent.store_transition(state, action, next_state, reward, done)
        agent.update_network()

        # Move to the next state
        state = next_state

        if done:
            if reward > 0:
                wins += 1

        i += 1
    cumulative_rewards.append(cum_reward)
    #plot_rewards(cumulative_rewards)
    logging.info("Episode lasted for %i time steps", i)

    # Update the target network, copying all weights and biases in DQN
    # Uncomment for Task 4
    if ep % TARGET_UPDATE == 0:
        agent.update_target_network()

    if ep % 10 == 0:
        logging.info("trained for: %s episodes", ep)
        logging.info("victory rate: %r", wins/(ep+1))

    # Save the policy
    # Uncomment for Task 4
    if ep % 1000 == 0:
        logging.info("saving model at ep: %s", ep)
        torch.save(agent.policy_net.state_dict(), "weights_%s_%d.mdl" % ("DQN", ep))

print('Complete')

