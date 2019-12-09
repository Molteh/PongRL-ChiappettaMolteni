import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import namedtuple

from test_agents.ActorCritic_SupervisedPreProcessing.models import Policy
from test_agents.ActorCritic_SupervisedPreProcessing.memory import ReplayMemory, Transition
from Pong_NN import PongNN as PNN


class Agent(object):

    def __init__(self):

        self.name = "GIORGIA SSJ"

        # TODO: Replay memory

        self.memory = ReplayMemory(50000)
        self.batch_size = 16
        self.replay_ratio = 4
        self.max_replay_size = 200
        self.action_probs = []
        self.rewards = []
        self.values = []

        # TODO: ACER from positions estimate
        self.train_device = "cpu"
        self.policy = None
        self.optimizer = None
        self.gamma = 0.98
        self.prev_obs = None

        # TODO: Supervised learning policies to estimate positions from raw image
        self.NN_ball_x = PNN()
        self.NN_ball_y = PNN()
        self.NN_my_y = PNN()
        self.NN_opponent_y = PNN()
        self.prev_ball_y = None

    def get_action(self, observation):

        def normalize_y(val):
            # First, clamp it to screen bounds
            y_min = 35
            y_max = 235
            val = np.clip(val, y_min, y_max)
            # Then, normalize to -1:1 range
            val = (val-y_min) / (y_max-y_min) * 2 - 1
            return val

        # TODO: Preprocess frame to reduce dimensionality and emphasize paddles/ball over background
        observation = self._preprocess(observation)

        # TODO: Create observation tensor
        observation = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Predict state variables
        my_y = normalize_y(self.NN_my_y(observation).detach().numpy()[0][0])
        opponent_y = normalize_y(self.NN_opponent_y(observation).detach().numpy()[0][0])
        ball_y = normalize_y(self.NN_ball_y(observation).detach().numpy()[0][0])

        if self.prev_ball_y is None:
            self.prev_ball_y = ball_y

        # TODO: Create approximated positions from supervised predictions
        positions = np.array([my_y, opponent_y, ball_y, self.prev_ball_y])

        # TODO: Store ball predictions for next observation
        self.prev_ball_y = ball_y

        # TODO: Create positions tensor
        positions = torch.from_numpy(positions).float().to(self.train_device)

        # TODO: Forward positions through the policy network -> Actor provides policy actions dist, Critic provides state value prediction
        dist, value = self.policy.forward(positions)

        # TODO: Get best action from probability distribution
        action = torch.argmax(dist.probs)

        return action

    def reset(self):
        self.prev_obs = None
        self.prev_ball_y = None

    def get_name(self):
        return self.name

    def load_model(self):

        # TODO: Actor Critic policy and optimizer weights to evaluate or resume training
        self.policy = Policy(4, 3).to(self.train_device)
        weights = torch.load("model.mdl", map_location=torch.device("cpu"))
        self.policy.load_state_dict(weights, strict=False)

        # TODO: Supervised learning policies weights used to preprocess image into positions
        ball_x_weights = torch.load("weights_XNN.mdl", map_location=torch.device("cpu"))
        self.NN_ball_x.load_state_dict(ball_x_weights, strict=False)
        ball_y_weights = torch.load("weights_YNN.mdl", map_location=torch.device("cpu"))
        self.NN_ball_y.load_state_dict(ball_y_weights, strict=False)
        my_y_weights = torch.load("weights_myYNN.mdl", map_location=torch.device("cpu"))
        self.NN_my_y.load_state_dict(my_y_weights, strict=False)
        opponent_y_weights = torch.load("weights_oppYNN.mdl", map_location=torch.device("cpu"))
        self.NN_opponent_y.load_state_dict(opponent_y_weights, strict=False)

    def train(self, env, opponent, resume=False, print_things=True, train_episodes=100000):

        # TODO: Set policy network and optimizer according to environment dimensionalities
        if not resume:
            obs_space_dim = env.observation_space.shape[-1]-2
            act_space_dim = env.action_space.n
            self.policy = Policy(obs_space_dim, act_space_dim).to(self.train_device)

        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=5e-3)
        self.policy.eval()

        # TODO: Set arrays to keep track of rewards and then plot them
        reward_history, timestep_history = [], []
        average_reward_history = []

        # # TODO: Run actual training
        for episode_number in range(train_episodes):
            reward_sum = 0
            timesteps = 0
            done = False
            # Reset the environment and observe the initial state
            observation, opponent_obs = env.reset()

            # Loop until the episode is over
            while not done:
                # Get action from the agent and store state - action prob - value
                action, action_prob, value = self._get_action_train(observation)

                # Get action from opponent
                opponent_action = opponent.get_action(opponent_obs)

                # Perform the action on the environment, get new state and reward
                (observation, opponent_obs), (reward, opponent_rew), done, info = env.step((action.detach().numpy(), opponent_action))

                # Store action's outcome (so that the agent can improve its policy)
                self._store_transition(action_prob, reward, value)

                # Store total episode reward
                reward_sum += reward
                timesteps += 1

            if print_things:
                print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                      .format(episode_number, reward_sum, timesteps))

            # Bookkeeping (mainly for generating plots)
            reward_history.append(reward_sum)
            timestep_history.append(timesteps)
            if episode_number > 100:
                avg = np.mean(reward_history[-100:])
            else:
                avg = np.mean(reward_history)
            average_reward_history.append(avg)

            # ACER updates at the end of the episode
            self._episode_finished()
            for trajectory_count in range(np.random.poisson(self.replay_ratio)):
                self._learning_iteration()

            if episode_number > 0 and episode_number % 1000 == 0:
                plt.plot(reward_history)
                plt.plot(average_reward_history)
                plt.legend(["Reward", "100-episode average"])
                plt.title("Reward history")
                plt.show()
                torch.save(self.policy.state_dict(), "model_{}.mdl".format(episode_number))

        # Training is finished - plot rewards
        if print_things:
            plt.plot(reward_history)
            plt.plot(average_reward_history)
            plt.legend(["Reward", "100-episode average"])
            plt.title("Reward history")
            plt.show()
            print("Training finished.")

        torch.save(self.policy.state_dict(), "model.mdl")

    def _get_action_train(self, observation, evaluation=False):

        # TODO: Preprocess observation to use only vertical positions
        processed_observation = np.array((observation[0], observation[1], observation[3], observation[5]))

        # TODO: Create observation tensor
        x = torch.from_numpy(processed_observation).float().to(self.train_device)

        # TODO: Forward positions through the policy network -> Actor provides policy actions dist, Critic provides state value prediction
        dist, value = self.policy.forward(x)  # Train using states

        # TODO: Return max if evaluation, else sample from the distribution returned by the policy
        if evaluation:
            action = torch.argmax(dist.probs)
        else:
            action = dist.sample()

        # TODO: Calculate the log probability of the action
        act_log_prob = dist.log_prob(action)

        return action, act_log_prob, value

    def _episode_finished(self):

        # TODO: Save trajectory to replay memory and reset temporary containers
        trajectory = self.action_probs, self.rewards, self.values
        self.memory.push(trajectory)
        self.action_probs = []
        self.rewards = []
        self.values = []

    def _learning_iteration(self):

        # TODO: Sample from replay memory
        batch_size = min(len(self.memory.memory), self.batch_size)
        trajectory = self.memory.sample(batch_size)
        action_probs = torch.stack(tuple(trajectory[0][0][0]), dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(trajectory[0][0][1], dim=0).to(self.train_device).squeeze(-1)
        values = torch.stack(trajectory[0][0][2], dim=0).to(self.train_device).squeeze(-1)

        # TODO: Compute discounted rewards and normalize
        # rewards = self._discount_rewards(rewards, gamma=self.gamma)
        if torch.sum(rewards) > 0:
            rewards = (rewards - torch.mean(rewards))/(torch.std(rewards)+1e-8)

        # TODO: Compute advantages
        advantages = rewards - values

        # TODO: Compute loss
        loss = torch.sum(-action_probs * advantages.detach())
        actor_loss = loss.mean()
        critic_loss = advantages.pow(2).mean()
        actor_critic_loss = actor_loss + critic_loss

        # TODO: Compute the gradients of loss w.r.t. network parameters
        actor_critic_loss.backward(retain_graph=True)

        # TODO: Update network parameters
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _store_transition(self, action_prob, reward, value):
        reward = torch.tensor([reward], dtype=torch.float32)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)
        self.values.append(value)

    def _discount_rewards(self, r, gamma):
        discounted_r = torch.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size(-1))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def _preprocess(self, frame):
        frame = frame[::2, ::2, 0]  # down sample by factor of 2
        frame[frame == 43] = 0  # erase background (background type 1)

        frame[frame != 0] = 1  # everything else (paddles, ball) just set to 1
        return frame.astype(np.float).ravel()
