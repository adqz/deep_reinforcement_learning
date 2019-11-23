import numpy as np
import torch
import torch.nn as nn
import gym
import pybulletgym

import copy, time
from tqdm import trange

from models import Actor, DecompCritic
from buffers import ReplayBuffer

class TD4:

    def __init__(self,
                 env,
                 sub_states,
                 layers,
                 gamma = 0.99,
                 tau = 1e-3,
                 pol_lr = 1e-4,
                 q_lr = 5e-3,
                 batch_size = 64,
                 buffer_size = 10000,
                 target_noise = 0.2,
                 action_noise = 0.1,
                 clip_range = 0.5,
                 update_delay = 2,
                 ):

        # environment stuff
        self.env = env
        self.num_act = env.action_space.shape[0]
        self.num_obs = env.observation_space.shape[0]
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
        self.target_noise = target_noise
        self.action_noise = action_noise
        self.clip_range = clip_range
        self.update_delay = update_delay

        # networks
        self.pol = Actor(self.num_obs, self.num_act, [400, 300]).double()
        self.q1 = DecompCritic(self.sub_states, self.num_act, layers).double()
        self.q2 = DecompCritic(self.sub_states, self.num_act, layers).double()
        self.pol.init_weights()
        self.q1.init_weights()
        self.q2.init_weights()

        # target networks
        self.target_pol = copy.deepcopy(self.pol).double()
        self.target_q1 = copy.deepcopy(self.q1).double()
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

    def count_parameters(self):
        ''' Count number of parameters in network and print them '''
        params_pol = sum(p.numel() for p in self.pol.parameters() if p.requires_grad)
        params_q1 = sum(p.numel() for p in self.q1.parameters() if p.requires_grad)
        params_q2 = sum(p.numel() for p in self.q2.parameters() if p.requires_grad)
        params_target_pol = sum(p.numel() for p in self.target_pol.parameters() if p.requires_grad)
        params_target_q1 = sum(p.numel() for p in self.target_q1.parameters() if p.requires_grad)
        params_target_q2 = sum(p.numel() for p in self.target_q2.parameters() if p.requires_grad)

        params_1 = (params_pol + params_q1 + params_q2)
        params_2 = (params_target_pol + params_target_q1 + params_target_q2)
        total_trainable_params = params_1 + params_2

        print(f'Parameter count: {params_1} + {params_2} = {total_trainable_params}')
        return

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
            self.count_parameters()
            iter_info.append((iter_reward, avg_q1_loss, avg_q2_loss, avg_obj))

        return iter_info


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
        self.num_obs = env.observation_space.shape[0]
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
        

        # target networks
        self.target_pol = copy.deepcopy(self.pol).double()
        self.target_q = copy.deepcopy(self.q).double()

        # Remove parameters of target network from computation graph
        self.target_pol = self.disable_gradient_calculation(self.target_pol)
        self.target_q = self.disable_gradient_calculation(self.target_q)

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


    def disable_gradient_calculation(self, model):
        ''' Sets requried_grad to False so gradients are not computed for the model '''
        print('Caution: Disbaling gradient')
        for p in model.parameters():
            p.requires_grad = False

        return model

    # update neural net
    def update_networks(self):
        # (pre_obs, action, reward, obs, done)
        pre_obs = torch.tensor(self.batch[0], dtype=torch.double)
        actions = torch.tensor(self.batch[1], dtype=torch.double)
        rewards = torch.tensor(self.batch[2], dtype=torch.double)
        obs = torch.tensor(self.batch[3], dtype=torch.double)
        done = torch.tensor(self.batch[4], dtype=torch.double).unsqueeze(1)

        self.q_opt.zero_grad()
        a_target = self.target_pol(obs).detach()
        q_target = self.target_q(obs, a_target).detach()
        y = rewards + (self.gamma * (1.0 - done) * q_target)
        # loss = torch.sum((y - self.q(pre_obs, actions)) ** 2) / self.batch_size
        loss = self.mse_loss(self.q(pre_obs, actions), y)
        loss.backward()
        self.q_opt.step()
        self.cum_loss += loss.item()

        self.pol_opt.zero_grad()
        objective = -self.q(pre_obs, self.pol(pre_obs)).mean()
        objective.backward()
        self.pol_opt.step()
        self.cum_obj += objective.item()


    # update target networks with tau
    def update_target_networks(self):
        for target, actual in zip(self.target_q.named_parameters(), self.q.named_parameters()):
            target[1].data.copy_(self.tau * actual[1].data + (1 - self.tau) * target[1].data)
        for target, actual in zip(self.target_pol.named_parameters(), self.pol.named_parameters()):
            target[1].data.copy_(self.tau * actual[1].data + (1 - self.tau) * target[1].data)

        # # Adnan's approach
        # for target, actual in zip(self.target_q.parameters(), self.q.parameters()):
        #     with torch.no_grad():
        #         target = self.tau * actual + ((1 - self.tau) * target)
        # for target, actual in zip(self.target_pol.parameters(), self.pol.parameters()):
        #     with torch.no_grad():
        #         target = self.tau * actual + ((1 - self.tau) * target)

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

    def count_trainable_parameters(self):
        ''' Count number of trainable parameters in network and print them '''
        params_pol = sum(p.numel() for p in self.pol.parameters() if p.requires_grad)
        params_q = sum(p.numel() for p in self.q.parameters() if p.requires_grad)
        params_target_pol = sum(p.numel() for p in self.target_pol.parameters() if p.requires_grad)
        params_target_q = sum(p.numel() for p in self.target_q.parameters() if p.requires_grad)

        params_1 = (params_pol + params_q)
        params_2 = (params_target_pol + params_target_q)
        total_trainable_params = params_1 + params_2

        print(f'Trainable parameter count: {params_1} + {params_2} = {total_trainable_params}')
        return

    def count_all_parameters(self):
        ''' Count all parameters in network and print them '''
        params_pol = sum(p.numel() for p in self.pol.parameters())
        params_q = sum(p.numel() for p in self.q.parameters())
        params_target_pol = sum(p.numel() for p in self.target_pol.parameters())
        params_target_q = sum(p.numel() for p in self.target_q.parameters())

        params_1 = (params_pol + params_q)
        params_2 = (params_target_pol + params_target_q)
        total_trainable_params = params_1 + params_2

        print(f'All parameter count: {params_1} + {params_2} = {total_trainable_params}')
        return

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
            self.count_all_parameters()
            self.count_trainable_parameters()
            iter_info.append((iter_reward, avg_loss, avg_obj))

        return iter_info


if __name__ == "__main__":
    seed = 1995
    torch.manual_seed(seed)
    env = gym.make("ReacherPyBulletEnv-v0", sparse_reward=True, rand_init=False)
    sub_states = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 7, 8]]
    layers = [[128, 128] for i in range(3)]

    td4 = DDDPG(env, sub_states, layers, buffer_size=1e6)

    res = td4.train(2000000, 1000)
    import pickle
    import time

    pickle.dump(res, open('./data/info_{}'.format(time.time()), 'wb'))
    path = './data/policy_{}'.format(time.time())
    torch.save(td4.pol.state_dict(), path)
