import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from test_agents.ACVisual_Anto.models import Policy

class Agent(object):
    def __init__(self):
        self.train_device = "cpu"
        self.policy = None
        self.optimizer = None
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.values = []
        self.prev_obs = None

    def preprocess(self, observation):
        observation = observation[::2, ::2].mean(axis=-1)
        observation = np.expand_dims(observation, axis=-1)
        if self.prev_obs is None:
            self.prev_obs = observation
        stack_ob = np.concatenate((self.prev_obs, observation), axis=-1)
        stack_ob = torch.from_numpy(stack_ob).float().unsqueeze(0)
        stack_ob = stack_ob.transpose(1, 3)
        self.prev_obs = observation
        return stack_ob

    def train(self, env, opponent, resume=False, print_things=True, train_run_id=0, train_episodes=20000):

        # TODO: Set policy network and optimizer according to environment dimensionalities
        if not resume:
            act_space_dim = env.action_space.n
            self.policy = Policy(act_space_dim).to(self.train_device)
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
                action, action_probabilities = self.get_action_train(observation)

                # Get action from opponent
                opponent_action = opponent.get_action(opponent_obs)

                # Perform the action on the environment, get new state and reward
                (observation, opponent_obs), (reward, opponent_rew), done, info = env.step((action.detach().numpy(), opponent_action))

                # Store action's outcome (so that the agent can improve its policy)
                self.store_outcome(reward)

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

            # MC policy gradient update at the end of the episode
            self.episode_finished(episode_number)

        # Training is finished - plot rewards
        if print_things:
            plt.plot(reward_history)
            plt.plot(average_reward_history)
            plt.legend(["Reward", "100-episode average"])
            plt.title("Reward history")
            plt.show()
            print("Training finished.")
        data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                             "train_run_id": [train_run_id] * len(reward_history),
                             "ActorCritic": ["PG"] * len(reward_history),
                             "reward": reward_history})
        torch.save(self.policy.state_dict(), "model_{}.mdl".format(train_run_id))
        return data

    def get_action(self, observation):
        action, action_prob = self.get_action_train(observation)
        return action

    def get_action_train(self, observation):

        # TODO: Create observation tensor with preprocessing
        x = self.preprocess(observation).to(self.train_device)

        # TODO: Forward state through the policy network -> Actor provides policy actions dist, Critic provides state value prediction
        dist, value = self.policy.forward(x)

        # TODO: Return action with highest probability
        action = torch.argmax(dist.probs)

        # TODO: Calculate the log probability of the action
        act_log_prob = dist.log_prob(action)

        # TODO: Save state - action prob - value
        self.states.append(observation)
        self.action_probs.append(act_log_prob)
        self.values.append(value)

        return action, act_log_prob

    def reset(self):
        self.prev_obs = None

    def get_name(self):
        return "GIORGIA"

    def load_model(self):
        self.policy = Policy(3).to(self.train_device)
        weights = torch.load("model.mdl", map_location=torch.device("cpu"))
        self.policy.load_state_dict(weights, strict=False)

    def episode_finished(self, episode_number):

        # TODO: Stack action_probs, rewards and values arrays
        action_probs = torch.stack(self.action_probs, dim=0) \
                .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        values = torch.stack(self.values, dim=0).to(self.train_device).squeeze(-1)

        # TODO: Reset values for next episode
        self.states, self.action_probs, self.rewards, self.values = [], [], [], []

        # TODO: Compute discounted rewards and normalize
        rewards = self.discount_rewards(rewards, gamma=self.gamma)
        rewards = (rewards - torch.mean(rewards))/torch.std(rewards)

        # TODO: Compute advantages
        advantages = rewards - values

        # TODO: Compute loss
        loss = torch.sum(-action_probs * advantages.detach())
        actor_loss = loss.mean()
        critic_loss = advantages.pow(2).mean()
        actor_critic_loss = actor_loss + critic_loss

        # TODO: Compute the gradients of loss w.r.t. network parameters
        actor_critic_loss.backward()

        # TODO: Update network parameters
        self.optimizer.step()
        self.optimizer.zero_grad()

    def store_outcome(self, reward):
        self.rewards.append(torch.Tensor([reward]))

    def discount_rewards(self, r, gamma):
        discounted_r = torch.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size(-1))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r
