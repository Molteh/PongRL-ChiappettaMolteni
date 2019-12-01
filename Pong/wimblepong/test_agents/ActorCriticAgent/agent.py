import numpy as np
import pickle
import copy

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I):
    #  """ prepro 200x235x3 uint8 frame into 10000 (100x100) 1D float vector """
    """ prepro 200x235x3 uint8 frame into 8300 (83x100) 1D float vector """
    I = I[35:200]  # crop - remove 35px from start & 35px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 43] = 0 # erase background (background type 1)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


class Agent(object):
    def __init__(self):
        self.H = 200  # number of hidden layer neurons
        self.D = 83 * 100  # input dimensionality: 83x100 grid
        #  self.D = 100 * 100  # input dimensionality: 100x100 grid
        self.prev_x = None
        self.model = {}
        self.model_target = {}
        self.init_model()
        self.name = "ActorCriticAgent"
        self.rewards = []
        self.model_file = "save.ac"

    def get_name(self):
        return self.name

    def load_model(self):
        self.model, self.model_target = pickle.load(open(self.model_file, 'rb'))

    def init_model(self):
        self.model.clear()
        self.model_target.clear()
        self.model['W1_policy'] = np.random.randn(self.H, self.D) / np.sqrt(self.D)  # "Xavier" initialization
        self.model['b1_policy'] = np.random.randn(self.H) / np.sqrt(4 * self.H)
        self.model['W2_policy'] = np.random.randn(self.H) / np.sqrt(self.H)
        self.model['b2_policy'] = 0.0
        self.model['W1_value'] = np.random.randn(self.H, self.D) / np.sqrt(self.D)  # "Xavier" initialization
        self.model['b1_value'] = np.random.randn(self.H) / np.sqrt(4 * self.H)
        self.model['W2_value'] = np.random.randn(self.H) / np.sqrt(self.H)
        self.model['b2_value'] = 0.0
        self.model_target = copy.deepcopy(self.model)

    def discount_rewards(self, r, gamma):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0:
                running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return np.array(discounted_r, dtype=np.float64)

    def policy_forward(self, x, model_type, model):
        h = np.dot(model['W1_' + model_type], x) + model['b1_' + model_type]
        h[h < 0] = 0  # ReLU nonlinearity
        out = np.dot(model['W2_' + model_type], h) + model['b2_' + model_type]
        if model_type == 'policy':
            out = sigmoid(out)
        return out, h  # return probability of taking action 2, and hidden state

    def policy_backward(self, eph, epx, epdlogp, model_type):
        """ backward pass. (eph is array of intermediate hidden states) """
        db2 = sum(epdlogp)[0]
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.model['W2_' + model_type])
        dh[eph <= 0] = 0  # backpro prelu
        db1 = sum(dh)
        dW1 = np.dot(dh.T, epx)
        return {'W1_' + model_type: dW1, 'W2_' + model_type: dW2, 'b1_' + model_type: db1, 'b2_' + model_type: db2}

    def get_action(self, observation):
        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.D)
        self.prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        aprob, _ = self.policy_forward(x, 'policy', self.model)
        action = 1 if np.random.uniform() < aprob else 2  # roll the dice!
        return action

    def reset(self):
        # Reset previous observation
        self.prev_x = None

    def train(self, env, opponent, batch_size, learning_rate, gamma, decay_rate, mom_rate, td_step, gamma_power, shrink_step, rmsprop, render):
        grad_buffer = {k: np.zeros_like(v) for k, v in
                       self.model.items()}  # update buffers that add up gradients over a batch
        momentum = {k: np.zeros_like(v) for k,v in self.model.items()}
        rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.items()}  # rmsprop memory

        obs1, obs2 = env.reset()
        prev_x = None  # used in computing the difference frame
        xs, h_ps, h_vs, dlogps, vs, tvs, dvs = [],[],[],[],[],[],[]
        running_reward = None
        reward_sum = 0
        episode_number = 0
        
        while episode_number < 30000:
            if render:
                env.render()

            # preprocess the observation, set input to network to be difference image
            cur_x = prepro(obs1)
            x = cur_x - prev_x if prev_x is not None else np.zeros(self.D)
            prev_x = cur_x

            # forward the policy network and sample an action from the returned probability
            aprob, h_p = self.policy_forward(x, 'policy', self.model)
            action1 = 1 if np.random.uniform() < aprob else 2  # roll the dice!

            action2 = opponent.get_action(obs2)

            v, h_v = self.policy_forward(x, 'value', self.model)
            tv, _ = self.policy_forward(x, 'value', self.model_target)
            # record various intermediates (needed later for backprop)
            xs.append(x)  # observation
            h_ps.append(h_p)  # hidden state
            h_vs.append(h_v)
            vs.append(v)
            tvs.append(tv)
            y = 1 if action1 == 1 else 0  # a "fake label"
            dlogps.append(
                y - aprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

            # step the environment and get new measurements
            (obs1, obs2), (rew1, rew2), done, info = env.step((action1, action2))
            reward_sum += rew1



            if done:  # an episode finished
                episode_number += 1

                if shrink_step and episode_number % 10000 == 0:
                    if td_step > 15:
                        td_step -= 1
                # calcutate td error
                dvs = [0] * len(vs)
                for i in range(len(vs)):
                    if len(vs) - 1 - i < td_step:
                        dvs[i] = rew1 * (gamma_power[len(vs) - 1 - i]) - vs[i]
                    else:
                        dvs[i] = gamma_power[td_step] * tvs[i + td_step] - vs[i]

                # stack together all inputs, hidden states, action gradients, and td for this episode
                epx = np.vstack(xs)
                eph_p = np.vstack(h_ps)
                eph_v = np.vstack(h_vs)
                epdlogp = np.vstack(dlogps)
                epv = np.vstack(dvs)
                xs, h_ps, h_vs, dlogps, vs, tvs, dvs = [], [], [], [], [], [], []  # reset array memory

                # discounted_epv = epv * np.vstack([gamma**i for i in range(len(epv))])
                epdlogp *= epv  # modulate the gradient with advantage (PG magic happens right here.)
                grad_p = self.policy_backward(eph_p, epx, epdlogp, 'policy')
                grad_v = self.policy_backward(eph_v, epx, epv, 'value')
                grad = dict(grad_p, **grad_v)

                for k in self.model:
                    grad_buffer[k] += grad[k]  # accumulate grad over batch

                if episode_number % batch_size == 0:
                    for k, v in self.model.items():
                        g = grad_buffer[k]  # gradient
                        if rmsprop:
                            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                            momentum[k] = mom_rate * momentum[k] + learning_rate * g / (
                                        np.sqrt(rmsprop_cache[k]) + 1e-5)
                        else:
                            momentum[k] = mom_rate * momentum[k] + learning_rate * g
                        self.model[k] += momentum[k]
                        grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

                        if 'value' in k:
                            self.model_target[k] = mom_rate * self.model_target[k] + (1 - mom_rate) * self.model[k]

                # boring book-keeping
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print('Resetting env. episode reward total was {}. running mean: {}'.format(reward_sum, running_reward))
                if episode_number % 100 == 0:
                    pickle.dump((self.model, self.model_target), open('save.ac', 'wb'))
                reward_sum = 0
                obs1, obs2 = env.reset()  # reset env
                prev_x = None

            if rew1 != 0:  # Pong has either +1 or -1 reward exactly when game ends.
                print('ep {}: game finished, reward: {}'.format(episode_number, rew1))
                if rew1 > 0:
                    print("YES")

