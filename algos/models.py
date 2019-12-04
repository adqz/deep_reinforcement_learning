import numpy as np
import torch
import torch.nn as nn

from itertools import chain

class Actor(torch.nn.Module):

    def __init__(self, state_size, action_size, layers):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.layers = layers

        self.fc1 = nn.Linear(state_size, layers[0])
        # self.bn1 = nn.BatchNorm1d(num_features=layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        # self.bn2 = nn.BatchNorm1d(num_features=layers[1])
        self.fc3 = nn.Linear(layers[1], action_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

    def init_weights(self):
        nn.init.uniform_(self.fc1.weight.data, -1 / (np.sqrt(self.state_size)), 1 / (np.sqrt(self.state_size)))
        nn.init.uniform_(self.fc2.weight.data, -1 / (np.sqrt(self.layers[0])), 1 / (np.sqrt(self.layers[0])))
        nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)
        nn.init.uniform_(self.fc1.bias.data, -1 / (np.sqrt(self.state_size)), 1 / (np.sqrt(self.state_size)))
        nn.init.uniform_(self.fc2.bias.data, -1 / (np.sqrt(self.layers[0])), 1 / (np.sqrt(self.layers[0])))
        nn.init.uniform_(self.fc3.bias.data, -3e-3, 3e-3)


class Critic(torch.nn.Module):

    def __init__(self, state_size, action_size, layers):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.layers = layers

        self.fc1 = nn.Linear(state_size + action_size, layers[0])
        # self.bn1 = nn.BatchNorm1d(num_features=layers[0])
        self.fc2 = nn.Linear(layers[0] , layers[1])
        self.fc3 = nn.Linear(layers[1], 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def init_weights(self):
        nn.init.uniform_(self.fc1.weight.data, -1 / (np.sqrt(self.state_size+ self.action_size)), 1 / (np.sqrt(self.state_size+ self.action_size)))
        nn.init.uniform_(self.fc2.weight.data, -1 / (np.sqrt(self.layers[0])), 1 / (np.sqrt(self.layers[0])))
        nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)
        nn.init.uniform_(self.fc1.bias.data, -1 / (np.sqrt(self.state_size+ self.action_size)), 1 / (np.sqrt(self.state_size+ self.action_size)))
        nn.init.uniform_(self.fc2.bias.data, -1 / (np.sqrt(self.layers[0])), 1 / (np.sqrt(self.layers[0])))
        nn.init.uniform_(self.fc3.bias.data, -3e-3, 3e-3)


class DecompCritic(torch.nn.Module):

    def __init__(self, sub_states, action_size, layers):
        super(DecompCritic, self).__init__()
        self.sub_critics = []
        for i, inds in enumerate(sub_states):
            sub_critic = Critic(len(inds), 1, layers[i]).double()
            sub_critic.init_weights()
            self.sub_critics.append(sub_critic)

        self.fc1 = nn.Linear(len(sub_states), layers[-1][0])
        self.fc2 = nn.Linear(layers[-1][0], layers[-1][1])
        self.fc3 = nn.Linear(layers[-1][1], 1)
        self.relu = nn.ReLU()

        self.sub_states = sub_states
        self.action_size = action_size
        self.layers = layers

    def forward(self, state, action):
        sub_q_values = []
        for i, ind in enumerate(self.sub_states):
            sub_states = state[:, ind]
            sub_q_values.append(self.sub_critics[i].forward(sub_states, action[:, i].unsqueeze(1)))

        x = torch.cat(sub_q_values, 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def init_weights(self):
        nn.init.uniform_(self.fc1.weight.data, -1 / (np.sqrt(len(self.sub_states))), 1 / (np.sqrt(len(self.sub_states))))
        nn.init.uniform_(self.fc2.weight.data, -1 / (np.sqrt(self.layers[-1][0])), 1 / (np.sqrt(self.layers[-1][0])))
        nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)
        nn.init.uniform_(self.fc1.bias.data, -1 / (np.sqrt(len(self.sub_states))), 1 / (np.sqrt(len(self.sub_states))))
        nn.init.uniform_(self.fc2.bias.data, -1 / (np.sqrt(self.layers[-1][0])), 1 / (np.sqrt(self.layers[-1][0])))
        nn.init.uniform_(self.fc3.bias.data, -3e-3, 3e-3)

    # def parameters(self, recurse: bool = ...) -> Iterator[Parameter]:
    def parameters(self, recurse = True):
        gen = chain(super().parameters())
        for i, ind in enumerate(self.sub_states):
            gen = chain(gen, self.sub_critics[i].parameters())
        return gen

    def named_parameters(self,  prefix='', recurse=True):
        gen = chain(super().named_parameters())
        for i, ind in enumerate(self.sub_states):
            gen = chain(gen, self.sub_critics[i].named_parameters())
        return gen

class CompCritic(torch.nn.Module):

    def __init__(self, state_size, layers):
        super(CompCritic, self).__init__()
        self.state_size = state_size
        self.layers = layers

        self.fc1 = nn.Linear(state_size, layers[0])
        # self.bn1 = nn.BatchNorm1d(num_features=layers[0])
        self.fc2 = nn.Linear(layers[0] , layers[1])
        self.fc3 = nn.Linear(layers[1], 1)
        self.relu = nn.ReLU()

    def forward(self, q):
        x = self.relu(self.fc1(q))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def init_weights(self):
        nn.init.uniform_(self.fc1.weight.data, -1 / (np.sqrt(self.state_size)), 1 / (np.sqrt(self.state_size)))
        nn.init.uniform_(self.fc2.weight.data, -1 / (np.sqrt(self.layers[0])), 1 / (np.sqrt(self.layers[0])))
        nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)
        nn.init.uniform_(self.fc1.bias.data, -1 / (np.sqrt(self.state_size)), 1 / (np.sqrt(self.state_size)))
        nn.init.uniform_(self.fc2.bias.data, -1 / (np.sqrt(self.layers[0])), 1 / (np.sqrt(self.layers[0])))
        nn.init.uniform_(self.fc3.bias.data, -3e-3, 3e-3)


class SumCritic(torch.nn.Module):

    def __init__(self, sub_states, action_size, layers):
        super(SumCritic, self).__init__()
        self.sub_critics = []
        for i, inds in enumerate(sub_states):
            sub_critic = Critic(len(inds), 1, layers[i]).double()
            sub_critic.init_weights()
            self.sub_critics.append(sub_critic)

        self.sub_states = sub_states
        self.action_size = action_size
        self.layers = layers

    def forward(self, state, action):
        sub_q_values = []
        for i, ind in enumerate(self.sub_states):
            sub_states = state[:, ind]
            sub_q_values.append(self.sub_critics[i].forward(sub_states, action[:, i].unsqueeze(1)))

        x = torch.sum(sub_q_values).unsqueeze(0)

        return x

    def init_weights(self):
        pass