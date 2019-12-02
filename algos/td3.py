import numpy as np
import torch
import torch.nn as nn
import gym
import pybulletgym

import copy, time
from tqdm import trange

from models import Actor, Critic
from buffers import ReplayBuffer

class TD3:

    def __init__(self,
                 env,
                 gamma = 0.99,
                 tau = 1e-3,
                 pol_lr = 1e-4,
                 q_lr = 5e-3,
                 batch_size = 64,
                 buffer_size = 10000,
                 target_noise = 0.2,
                 action_noise = 0.1,
                 clip_range = 0.5,
                 update_delay = 2
                 ):

        # environment stuff
        self.env = env
        self.num_act = env.action_space.shape[0]
        self.num_obs = env.observation_space.shape[0]
        self.eval_env = copy.deepcopy(env)
        self.eval_env.rand_init = False

        # hyper parameters
        self.gamma = gamma
        self.tau = tau
        self.pol_lr = pol_lr
        self.q_lr = q_lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_noise = target_noise
        self.action_noise = action_noise
        self.clip_range = clip_range
        self.update_delay = 2

        # networks
        self.pol = Actor(self.num_obs, self.num_act, [400, 300]).double()
        self.q1 = Critic(self.num_obs, self.num_act, [400, 300]).double()
        self.q2 = Critic(self.num_obs, self.num_act, [400, 300]).double()
        self.pol.init_weights()
        self.q1.init_weights()
        self.q2.init_weights()
        self.target_pol = copy.deepcopy(self.pol).double()
        self.target_q1= copy.deepcopy(self.q1).double()
        self.target_q2 = copy.deepcopy(self.q2).double()

        # optimizers, buffer
        self.pol_opt = torch.optim.Adam(self.pol.parameters(),
                                        lr=self.pol_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(),
                                      lr=self.q_lr,)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(),
                                       lr=self.q_lr, )
        self.buffer = ReplayBuffer(self.buffer_size, 1000)
        self.mse_loss = torch.nn.MSELoss()

        self.cum_q1_loss = 0
        self.cum_q2_loss = 0
        self.cum_obj = 0

    def noise(self, noise, length):
        return torch.tensor(np.random.multivariate_normal(
                mean=np.array([0.0 for i in range(length)]),
                cov=np.diag([noise for i in range(length)])), dtype=torch.double)

    # fill up buffer first
    def prep_buffer(self):
        obs = self.env.reset()
        while not self.buffer.ready:
            pre_obs = obs
            action = self.env.action_space.sample()
            obs, reward, done, _ = self.env.step(action)
            self.buffer.insert((pre_obs, action, reward, obs, done))
            if done: obs = self.env.reset()

    # for clipping off values
    def clip(self, x, l, u):
        if isinstance(l, (list, np.ndarray)):
            lower = torch.tensor(l, dtype=torch.double)
            upper = torch.tensor(u, dtype=torch.double)
        elif isinstance(l, (int, float)):
            lower = torch.tensor([l for i in range(len(x))], dtype=torch.double)
            upper = torch.tensor([u for i in range(len(x))], dtype=torch.double)
        else:
            assert(False, "Clipped wrong")

        return torch.max(torch.min(x, upper), lower)

    # update neural net
    def update_networks(self):
        # (pre_obs, action, reward, obs, done)
        pre_obs = torch.tensor(self.batch[0], dtype=torch.double)
        actions = torch.tensor(self.batch[1], dtype=torch.double)
        rewards = torch.tensor(self.batch[2], dtype=torch.double)
        obs = torch.tensor(self.batch[3], dtype=torch.double)
        done = torch.tensor(self.batch[4], dtype=torch.double).unsqueeze(1)

        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        noise = self.clip(torch.tensor(self.noise(self.target_noise, self.num_act)),
                            -self.clip_range,
                            self.clip_range)
        target_action = self.clip(self.target_pol(obs) + noise,
                                    self.env.action_space.low,
                                    self.env.action_space.high)
        target_q1_val = self.target_q1(obs, target_action)
        target_q2_val = self.target_q2(obs, target_action)
        y = rewards + (self.gamma * (1.0 - done) * torch.min(target_q1_val, target_q2_val))
        # loss = torch.sum((y - self.q(pre_obs, actions)) ** 2) / self.batch_size
        q1_loss = self.mse_loss(self.q1(pre_obs, actions), y)
        q2_loss = self.mse_loss(self.q2(pre_obs, actions), y)
        q1_loss.backward(retain_graph=True)
        q2_loss.backward()
        self.q1_opt.step()
        self.q2_opt.step()
        self.cum_q1_loss += q1_loss.item()
        self.cum_q2_loss += q2_loss.item()

        self.pol_opt.zero_grad()
        objective = -self.q1(pre_obs, self.pol(pre_obs)).mean()
        objective.backward()
        self.pol_opt.step()
        self.cum_obj += objective.item()


    # update target networks with tau
    def update_target_networks(self):
        for target, actual in zip(self.target_q1.named_parameters(), self.q1.named_parameters()):
            target[1].data.copy_(self.tau * actual[1].data + (1 - self.tau) * target[1].data)
        for target, actual in zip(self.target_q2.named_parameters(), self.q2.named_parameters()):
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
                action = action + self.noise(self.action_noise, self.num_act)
                action = action.detach().numpy()
                obs, reward, done, _ = self.env.step(action)
                self.buffer.insert((pre_obs, action, reward, obs, done))
                if render:
                    self.env.render('human')
                    time.sleep(0.000001)
                if done: obs = self.env.reset()

                # TD3 updates less often
                if j % self.update_delay == 0:
                    # sample from buffer, train one step, update target networks
                    self.batch = self.buffer.sample(self.batch_size)
                    self.update_networks()
                    self.update_target_networks()


            iter_reward = self.policy_eval()
            avg_q1_loss = self.cum_q1_loss/((i+1)*eval_len)
            avg_q2_loss = self.cum_q2_loss / ((i + 1) * eval_len)
            avg_obj = self.cum_obj/((i+1)*eval_len)
            print("Iteration {}/{}".format((i+1)*eval_len, num_iters))
            print("Rewards: {} | Q Loss: {}, {} | Policy Objective: {}".format(iter_reward,
                                                                         avg_q1_loss,
                                                                         avg_q2_loss,
                                                                         avg_obj))
            iter_info.append((iter_reward, avg_q1_loss, avg_q2_loss, avg_obj))

        return iter_info


if __name__ == "__main__":
    torch.manual_seed(58008)
    env = gym.make("ReacherPyBulletEnv-v0")
    td3 = TD3(env, batch_size=64)
    info = td3.train(10000, 5000)

