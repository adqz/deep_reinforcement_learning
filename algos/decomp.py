import numpy as np
import torch
import torch.nn as nn
import gym

import copy, time
from tqdm import trange

from models import Actor, DecompCritic
from buffers import ReplayBuffer

class DDDPG:

    def __init__(self,
                 env,
                 sub_states,
                 layers,
                 gamma=0.99,
                 tau=1e-3,
                 pol_lr=1e-4,
                 q_lr=1e-3,
                 batch_size=64,
                 buffer_size=10000,
                 ):

        # environment stuff
        self.env = env
        self.num_act = env.action_space.shape[0]
        self.num_obs = env.observation_space.shape[0] - 1
        self.eval_env = copy.deepcopy(env)
        self.sub_states = sub_states
        self.layers = layers

        # hyper parameters
        self.gamma = gamma
        self.tau = tau
        self.pol_lr = pol_lr
        self.q_lr = q_lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # networks
        self.pol = Actor(self.num_obs, self.num_act, [400, 300]).double()
        # decomp critic
        self.q = DecompCritic(self.sub_states, self.num_act, layers).double()
        self.pol.init_weights()
        self.q.init_weights()
        self.target_pol = copy.deepcopy(self.pol).double()
        self.target_q = copy.deepcopy(self.q).double()

        # optimizers, buffer
        self.pol_opt = torch.optim.Adam(self.pol.parameters(),
                                        lr=self.pol_lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(),
                                      lr=self.q_lr,)
        self.buffer = ReplayBuffer(self.buffer_size, 1000)
        self.mse_loss = torch.nn.MSELoss()

        self.cum_loss = 0
        self.cum_obj = 0

    # fill up buffer first
    def prep_buffer(self):
        obs = self.env.reset()
        while not self.buffer.ready:
            pre_obs = obs
            action = self.env.action_space.sample()
            obs, reward, done, _ = self.env.step(action)
            self.buffer.insert((pre_obs, action, reward, obs, done))
            if done: obs = self.env.reset()


    # update neural net
    def update_networks(self):
        # (pre_obs, action, reward, obs, done)
        pre_obs = torch.tensor(self.batch[0], dtype=torch.double)
        actions = torch.tensor(self.batch[1], dtype=torch.double)
        rewards = torch.tensor(self.batch[2], dtype=torch.double)
        obs = torch.tensor(self.batch[3], dtype=torch.double)
        done = torch.tensor(self.batch[4], dtype=torch.double).unsqueeze(1)

        self.q_opt.zero_grad()
        y = rewards + (self.gamma * (1.0 - done) * self.target_q(obs, self.target_pol(obs)))
        # loss = torch.sum((y - self.q(pre_obs, actions)) ** 2) / self.batch_size
        loss = self.mse_loss(self.q(pre_obs, actions), y)
        loss.backward()
        self.q_opt.step()
        self.cum_loss += loss

        self.pol_opt.zero_grad()
        objective = -self.q(pre_obs, self.pol(pre_obs)).mean()
        objective.backward()
        self.pol_opt.step()
        self.cum_obj += objective


    # update target networks with tau
    def update_target_networks(self):
        for target, actual in zip(self.target_q.named_parameters(), self.q.named_parameters()):
            target[1].data.copy_(self.tau * actual[1].data + (1 - self.tau) * target[1].data)
        for target, actual in zip(self.target_pol.named_parameters(), self.pol.named_parameters()):
            target[1].data.copy_(self.tau * actual[1].data + (1 - self.tau) * target[1].data)

    def policy_eval(self):
        state = self.eval_env.reset()
        done = False
        rewards = []
        while not done:
            inp = torch.tensor(state, dtype=torch.double)
            action = self.pol(inp)
            action = action.detach().numpy()
            next_state, r, done, _ = self.eval_env.step(action)
            rewards.append(r)
            # self.eval_env.render()
            # time.sleep(0.1)
            state = next_state

        total = sum(rewards)
        return total

    def train(self, num_iters=200000, eval_len=1000, render=False):
        print("Start")
        if render: self.env.render('human')
        self.prep_buffer()
        obs = self.env.reset()
        iter_info = []

        # train for num_iters
        for i in range(int(num_iters/eval_len)):
            for j in trange(eval_len):
                # one step and put into buffer
                pre_obs = obs
                inp = torch.tensor(obs, dtype=torch.double)
                action = self.pol(inp)
                action = action.detach().numpy() + np.random.multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[0.1, 0.0], [0.0, 0.1]]))
                obs, reward, done, _ = self.env.step(action)
                self.buffer.insert((pre_obs, action, reward, obs, done))
                if render:
                    self.env.render('human')
                    time.sleep(0.000001)
                if done: obs = self.env.reset()

                # sample from buffer, train one step, update target networks
                self.batch = self.buffer.sample(self.batch_size)
                self.update_networks()
                self.update_target_networks()


            iter_reward = self.policy_eval()
            avg_loss = self.cum_loss/((i+1)*eval_len)
            avg_obj = self.cum_obj/((i+1)*eval_len)
            print("Iteration {}/{}".format((i+1)*eval_len, num_iters))
            print("Rewards: {} | Q Loss: {} | Policy Objective: {}".format(iter_reward,
                                                                         avg_loss,
                                                                         avg_obj))
            iter_info.append((iter_reward, avg_loss, avg_obj))

        return iter_info


if __name__ == "__main__":

    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=True)
    sub_states = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 6, 7]]
    layers = [[64, 64] for i in range(3)]

    dddpg = DDDPG(env, sub_states, layers)
    dddpg.train(50000, 1000)
