import argparse
import subprocess
from itertools import count
import torch
from tensorboard_logger import configure, log_value
import gym

import sys
import os

from test_agents.TRPOAgent.models import DQNRegressor, DQNSoftmax
from test_agents.TRPOAgent.agent import Agent

from wimblepong.simple_ai import SimpleAi

def main(env, opponent):
    policy_model = DQNSoftmax(env.action_space.n)
    value_function_model = DQNRegressor()
    agent = TRPOAgent(env, opponent, policy_model, value_function_model)

    subprocess.Popen(["tensorboard", "--logdir", "runs"])
    configure("runs/pong-run")

    for t in count():
        reward = agent.step()
        print('score', reward, t)
        log_value('score', reward, t)
        if t % 100 == 0:
            torch.save(policy_model.state_dict(), "model.mdl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID',
                        default='WimblepongVisualMultiplayer-v0')
    parser.add_argument('opponent_dir', default=None, nargs="?", help="opponent directory")
    args = parser.parse_args()

    env = gym.make(args.env)

    if args.opponent_dir:
        sys.path.insert(0, args.opponent_dir)
        import agent
        orig_wd = os.getcwd()
        os.chdir(args.opponent_dir)
        agent = agent.Agent()
        agent.load_model()
        os.chdir(orig_wd)
        del sys.path[0]
    else:
        agent = SimpleAi(env, player_id=2)

    main(env, agent)
