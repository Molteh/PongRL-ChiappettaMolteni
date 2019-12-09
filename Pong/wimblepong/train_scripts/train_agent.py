import argparse
import sys
import os
from matplotlib import font_manager
import importlib
import gym

from wimblepong.simple_ai import SimpleAi


parser = argparse.ArgumentParser()
parser.add_argument("dir1", type=str, help="Directory to agent 1 to be trained.")
parser.add_argument("dir2", type=str, default=None, nargs="?",
                    help="Directory to agent 2 to be used as opponent in agent 1 training. If empty, SimpleAI is used instead.")
parser.add_argument("--render", "-r", action="store_true", help="Render the competition.")
parser.add_argument("--games", "-g", type=int, default=100, help="number of games.")

args = parser.parse_args()

env = gym.make("WimblepongMultiplayer-v0")

sys.path.insert(0, args.dir1)
import agent
orig_wd = os.getcwd()
os.chdir(args.dir1)
agent1 = agent_acer.Agent()
os.chdir(orig_wd)
del sys.path[0]

if args.dir2:
    sys.path.insert(0, args.dir2)
    importlib.reload(agent)
    os.chdir(args.dir2)
    agent2 = agent.Agent()
    agent2.load_model()
    os.chdir(orig_wd)
    del sys.path[0]
else:
    agent2 = SimpleAi(env, player_id=2)

resume = False  # resume from previous checkpoint?

if resume:
    sys.path.insert(0, args.dir1)
    orig_wd = os.getcwd()
    os.chdir(args.dir1)
    agent1.load_model()
    print("Resuming from previous model")
    os.chdir(orig_wd)
    del sys.path[0]

agent1.train(env, agent2, resume)
