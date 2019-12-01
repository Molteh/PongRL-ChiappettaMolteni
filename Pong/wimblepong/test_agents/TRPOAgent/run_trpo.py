import argparse
import subprocess
from itertools import count
import torch
from tensorboard_logger import configure, log_value
import gym

from models import DQNRegressor, DQNSoftmax
from trpo_agent import TRPOAgent
from utils.atari_wrapper import make_atari, wrap_deepmind


def main(env_id, opponent):
  env = gym.make("WimblepongVisualMultiplayer-v0")
  policy_model = DQNSoftmax(env.action_space.n)
  value_function_model = DQNRegressor()
  agent1 = TRPOAgent(env, policy_model, value_function_model)
  agent2 = opponent

  subprocess.Popen(["tensorboard", "--logdir", "runs"])
  configure("runs/pong-run")

  for t in count():
    reward = agent1.step()
    log_value('score', reward, t)
    if t % 100 == 0:
      torch.save(policy_model.state_dict(), "policy_model.pth")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--env', help='environment ID',
                      default='WimblepongVisualMultiplayer-v0')
  args = parser.parse_args()
  main(args.env)
