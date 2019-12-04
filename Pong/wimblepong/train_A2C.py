import torch
import wimblepong
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt

from test_agents.A2CAgent.agent import Agent, Policy

# Make the environment
env = gym.make("WimblepongSimpleAI-v0")

import pandas as pd


# Policy training function
def train(env_name, print_things=True, train_run_id=0, train_episodes=5000):
    # Create a Gym environment
    env = gym.make(env_name)

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.n
    observation_space_dim = env.observation_space.shape[-1]

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, action_space_dim)

    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False

        rewards = []
        values = []
        log_probs = []
        entropy_term = 0
        episode_reward = 0

        # Reset the environment and observe the initial state
        state = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action and state value from the agent
            action, dist, value = agent.get_action(state)

            # Perform the action on the environment, get new state and reward
            new_state, reward, done, info = env.step(action)

            log_prob = torch.log(dist.squeeze(0)[action])
            entropy = -torch.sum(dist.mean() * torch.log(dist))

            rewards.append(reward)
            values.append(value.detach().numpy()[0])
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state
            episode_reward += reward

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        if print_things:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps) ({} entropy) ({} entropy term)"
                  .format(episode_number, reward_sum, timesteps, entropy, entropy_term))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        _, _, next_value = agent.get_action(state)
        agent.update(rewards, values, next_value, log_probs, entropy_term)

    # Training is finished - plot rewards
    if print_things:
        plt.plot(reward_history)
        plt.plot(average_reward_history)
        plt.legend(["Reward", "100-episode average"])
        plt.title("Reward history")
        plt.show()
        print("Training finished.")
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         # TODO: Change algorithm name for plots, if you want
                         "algorithm": ["PG"]*len(reward_history),
                         "reward": reward_history})
    torch.save(agent.ac_net.state_dict(), "model_%s_%d.mdl" % ("A2C", episode_number))
    return data


# Function to test a trained policy
def test(env_name, episodes, params, render):
    # Create a Gym environment
    env = gym.make(env_name)

    # Get dimensionalities of actions and observations
    action_space_dim = env.action_space.n
    observation_space_dim = env.observation_space.shape[-1]

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(params)
    agent = Agent(policy)

    test_reward, test_len = 0, 0
    for ep in range(episodes):
        done = False
        observation = env.reset()
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, dist, value = agent.get_action(observation, evaluation=True)
            observation, reward, done, info = env.step(action.detach().cpu().numpy())

            log_prob = torch.log(dist.squeeze(0)[action])
            entropy = -torch.sum(dist.mean() * torch.log(dist))

            if render:
                env.render()
            test_reward += reward
            test_len += 1
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None, help="Model to be tested")
    parser.add_argument("--env", type=str, default="WimblepongSimpleAI-v0", help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=15000, help="Number of episodes to train for")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    args = parser.parse_args()

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.

    if args.test is None:
        try:
            train(args.env, train_episodes=args.train_episodes)
        # Handle Ctrl+C - save model and go to tests
        except KeyboardInterrupt:
            print("Interrupted!")
    else:
        state_dict = torch.load(args.test)
        print("Testing...")
        test(args.env, 100, state_dict, args.render_test)

