import gym
import torch
import argparse
import time

from ddpg import *

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--trials",
                    type=int,
                    help="number of trials",
                    default=1)
parser.add_argument("render",
                    type=bool,
                    help="render or naw",
                    default=False)
parser.add_argument("path",
                    type=str,
                    help="path of policy",
                    default="no exist")
args = parser.parse_args()

env = gym.make("ReacherPyBulletEnv-v0", sparse_reward = True)

pol = Actor(env.observation_space.shape[0], env.action_space.shape[0], [400, 300]).double()
path = args.path
pol.load_state_dict(torch.load(path))
pol.eval()

if args.render: env.render('human')
for i in range(args.trials):
    state = env.reset()
    done = False
    rewards = []
    while not done:
        inp = torch.tensor(state, dtype=torch.double)
        action = pol(inp)
        action = action.detach().numpy()
        next_state, r, done, _ = env.step(action)
        rewards.append(r)
        if args.render:
            env.render('human')
            time.sleep(0.1)
        state = next_state

    total = sum(rewards)
    print("Reward: {}".format(total))