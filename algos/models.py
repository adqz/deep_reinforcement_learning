import numpy as np
import torch
import torch.nn as nn


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
        assert self.state_size in state.shape, 'Invalid dim! No {0} in {1}'.format(self.state_size, state.shape)
        assert self.action_size in action.shape, 'Invalid dim! No {0} in {1}'.format(self.action_size, action.shape)
        # debug

        x = torch.cat([state, action], 1)

        assert (self.state_size + self.action_size) in x.shape, \
        'Input size mismatch. Input shape: {0} does not have {1} in it'.format(x.shape, (self.state_size + self.action_size)) 

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
            sub_critic = Critic(len(inds), action_size, layers[i]).double()
            sub_critic.init_weights()
            self.sub_critics.append(sub_critic)

        self.fc1 = nn.Linear(len(sub_states)+action_size, layers[-1][0])
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
            # print('sub_states.shape, action.shape = ', sub_states.shape, action.shape) #debug
            sub_q_values.append(self.sub_critics[i].forward(sub_states, action))

        x = torch.cat(sub_q_values + [action], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def init_weights(self):
        nn.init.uniform_(self.fc1.weight.data, -1 / (np.sqrt(len(self.sub_states) + self.action_size)), 1 / (np.sqrt(len(self.sub_states)+ self.action_size)))
        nn.init.uniform_(self.fc2.weight.data, -1 / (np.sqrt(self.layers[-1][0])), 1 / (np.sqrt(self.layers[-1][0])))
        nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)
        nn.init.uniform_(self.fc1.bias.data, -1 / (np.sqrt(len(self.sub_states) + self.action_size)), 1 / (np.sqrt(len(self.sub_states)+ self.action_size)))
        nn.init.uniform_(self.fc2.bias.data, -1 / (np.sqrt(self.layers[-1][0])), 1 / (np.sqrt(self.layers[-1][0])))
        nn.init.uniform_(self.fc3.bias.data, -3e-3, 3e-3)