import numpy as np
import pickle

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 43] = 0 # erase background (background type 1)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


class Agent(object):
    def __init__(self):
        self.H = 200  # number of hidden layer neurons
        self.D = 100 * 100  # input dimensionality: 100x100 grid
        self.prev_x = None
        self.model = {}
        self.init_model()
        self.name = "PolicyGradientNN"
        self.rewards = []
        self.model_file = "save.p"

    def get_name(self):
        return self.name

    def load_model(self):
        self.model = pickle.load(open(self.model_file, 'rb'))

    def init_model(self):
        self.model.clear()
        self.model['W1'] = np.random.randn(self.H, self.D) / np.sqrt(self.D)
        self.model['W2'] = np.random.randn(self.H) / np.sqrt(self.H)

    def discount_rewards(self, r, gamma):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return np.array(discounted_r, dtype=np.float64)

    def policy_forward(self, x):
        h = np.dot(self.model['W1'], x)
        h[h < 0] = 0  # ReLU nonlinearity
        logp = np.dot(self.model['W2'], h)
        p = sigmoid(logp)
        return p, h  # return probability of taking action 2, and hidden state

    def policy_backward(self, epx, eph, epdlogp):
        """ backward pass. (eph is array of intermediate hidden states) """
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0  # backpro prelu
        dW1 = np.dot(dh.T, epx)
        return {'W1': dW1, 'W2': dW2}

    def get_action(self, observation):
        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.D)
        self.prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, _ = self.policy_forward(x)
        action = 1 if np.random.uniform() < aprob else 2  # roll the dice!
        return action

    def reset(self):
        # Reset previous observation
        self.prev_x = None

    def train(self, env, opponent, batch_size, learning_rate, gamma, decay_rate, render):
        grad_buffer = {k: np.zeros_like(v) for k, v in
                       self.model.items()}  # update buffers that add up gradients over a batch
        rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.items()}  # rmsprop memory

        obs1, obs2 = env.reset()
        prev_x = None  # used in computing the difference frame
        xs, hs, dlogps, drs = [], [], [], []
        running_reward = None
        reward_sum = 0
        episode_number = 0

        self.load_model()

        while episode_number < 30000:
            if render:
                env.render()

            # preprocess the observation, set input to network to be difference image
            cur_x = prepro(obs1)
            x = cur_x - prev_x if prev_x is not None else np.zeros(self.D)
            prev_x = cur_x

            # forward the policy network and sample an action from the returned probability
            aprob, h = self.policy_forward(x)
            action1 = 1 if np.random.uniform() < aprob else 2  # roll the dice!

            action2 = opponent.get_action(obs2)

            # record various intermediates (needed later for backprop)
            xs.append(x)  # observation
            hs.append(h)  # hidden state
            y = 1 if action1 == 1 else 0  # a "fake label"
            dlogps.append(
                y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

            # step the environment and get new measurements
            (obs1, obs2), (rew1, rew2), done, info = env.step((action1, action2))
            reward_sum += rew1

            drs.append(rew1)  # record reward (has to be done after we call step() to get reward for previous action)

            if done:  # an episode finished
                episode_number += 1

                # stack together all inputs, hidden states, action gradients, and rewards for this episode
                epx = np.vstack(xs)
                eph = np.vstack(hs)
                epdlogp = np.vstack(dlogps)
                epr = np.vstack(drs)
                xs, hs, dlogps, drs = [], [], [], []  # reset array memory

                # compute the discounted reward backwards through time
                discounted_epr = self.discount_rewards(epr, gamma)
                # standardize the rewards to be unit normal (helps control the gradient estimator variance)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)

                epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
                grad = self.policy_backward(epx, eph, epdlogp)
                for k in self.model:
                    grad_buffer[k] += grad[k]  # accumulate grad over batch

                # perform rmsprop parameter update every batch_size episodes
                if episode_number % batch_size == 0:
                    for k, v in self.model.items():
                        g = grad_buffer[k]  # gradient
                        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                        self.model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                        grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

                # boring book-keeping
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print
                'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
                if episode_number % 100 == 0:
                    pickle.dump(self.model, open('save.p', 'wb'))
                reward_sum = 0
                obs1, obs2 = env.reset()  # reset env
                prev_x = None

            if rew1 != 0:  # Pong has either +1 or -1 reward exactly when game ends.
                print('ep {}: game finished, reward: {}'.format(episode_number, rew1))
                if rew1 > 0:
                    print("YES")

