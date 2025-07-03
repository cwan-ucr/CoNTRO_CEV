import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import os


from torch.distributions import Normal
from utils.utils import plot_grad_flow, layer_init_filter


# Replay buffer
class Replay_Buffer:
    def __init__(self,
                 capacity,
                 state_size,
                 action_size,
                 batch_size,
                 device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.s = np.zeros((capacity, state_size), dtype=np.float32)
        self.a = np.zeros((capacity, action_size), dtype=np.float32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s_ = np.zeros((capacity, state_size), dtype=np.float32)
        self.d = np.zeros((capacity), dtype=np.bool_)

    def add(self, s, a, r, s_, d):

        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_[self.ptr] = s_
        self.d[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        ind = np.random.randint(0, self.size, size=(self.batch_size, ))

        return (torch.from_numpy(self.s[ind]).to(dtype=torch.float32, device=self.device),
                torch.from_numpy(self.a[ind]).to(dtype=torch.float32, device=self.device),
                torch.from_numpy(self.r[ind]).to(dtype=torch.float32, device=self.device),
                torch.from_numpy(self.s_[ind]).to(dtype=torch.float32, device=self.device),
                torch.from_numpy(self.d[ind]).to(dtype=torch.bool, device=self.device))

    def len(self):
        return self.size


# Optimizer for multi-processing
class Shared_Optimizer:
    def __init__(self,
                 actor_params,
                 critic_params,
                 alpha_params,
                 lr,
                 weight_decay):

        self.optim_actor = optim.Adam(actor_params,
                                      lr=lr,
                                      weight_decay=weight_decay)
        self.optim_critic = optim.Adam(critic_params,
                                       lr=lr,
                                       weight_decay=weight_decay)
        self.optim_alpha = optim.Adam([alpha_params],
                                      lr=lr)

        # Loss list
        self.actor_loss = []
        self.critic_loss = []
        self.alpha_loss = []




# Simple one layer
class ActorModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, action_size):
        super(ActorModel, self).__init__()
        self.name = 'actor'
        self.action_size = action_size
        self.training = True
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, action_size * 2))
        self.model = nn.Sequential(*layers)
        self.model.apply(layer_init_filter)

    def forward(self, hidden_states):
        x = self.model(hidden_states)

        if x.shape[0] == self.action_size * 2:
            x_mu = x[:self.action_size]
            x_log_std = torch.clamp(x[self.action_size:],
                                    min=-20,
                                    max=2)
        else:
            x_mu = x[:, :self.action_size]
            x_log_std = torch.clamp(x[:, :self.action_size],
                                    min=-20,
                                    max=2)
        x_std = torch.exp(x_log_std + 1e-6)

        if torch.any(torch.isnan(x_mu)):
            print("x_mu contains NaN values")
            x_mu = torch.nan_to_num(x_mu, nan=1e-3)  # 替换 NaN
        if torch.any(torch.isnan(x_std)):
            print("x_std contains NaN values")
            x_std = torch.nan_to_num(x_mu, nan=1.0) + 1e-6  # 确保标准差为正

        dist = Normal(x_mu, x_std)
        u = dist.rsample() if self.training else x_mu
        action = torch.tanh(u)
        log_prob = dist.log_prob(u)
        log_prob -= torch.log(1 - action.pow(2) + 1e-7)

        if log_prob.shape[0] == self.action_size:
            log_prob = log_prob.sum(dim=0, keepdim=True)
        else:
            log_prob = log_prob.sum(dim=1, keepdim=True)

        return action.to(dtype=torch.float32), log_prob.squeeze(-1)

# Double Q network
class CriticModel(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(CriticModel, self).__init__()
        self.name = 'critic'
        layers_1 = []
        layers_2 = []

        prev_size = input_size
        for h in hidden_sizes:
            layers_1.append(nn.Linear(prev_size, h))
            layers_1.append(nn.ReLU())
            prev_size = h
        layers_1.append(nn.Linear(prev_size, 1))

        prev_size = input_size
        for h in hidden_sizes:
            layers_2.append(nn.Linear(prev_size, h))
            layers_2.append(nn.ReLU())
            prev_size = h
        layers_2.append(nn.Linear(prev_size, 1))

        self.Q1 = nn.Sequential(*layers_1)
        self.Q2 = nn.Sequential(*layers_2)

        self.Q1.apply(layer_init_filter)
        self.Q2.apply(layer_init_filter)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.Q1(x), self.Q2(x)


class NN_SAC_Model(nn.Module):
    def __init__(self,
                 state_size,
                 action_size,
                 hidden_sizes,
                 alpha,
                 device):
        super(NN_SAC_Model, self).__init__()

        self.name = 'SAC_cycle_based_tsc'

        self.parent_path = os.path.dirname(__file__)

        self.actor_model = ActorModel(state_size, hidden_sizes, action_size).to(device)
        self.critic_model = CriticModel(state_size + action_size, hidden_sizes).to(device)

        self.target_critic_model = copy.deepcopy(self.critic_model)
        for params in self.target_critic_model.parameters():
            params.requires_grad = False

        # 定义adaptive temperature
        self.target_entropy = nn.Parameter(torch.tensor(-action_size, dtype=torch.float32)).to(device)
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha, dtype=torch.float32))).to(device)
        # self.target_entropy = torch.tensor(-action_size,
        #                                    dtype=torch.float32,
        #                                    requires_grad=True,
        #                                    device=device)
        # self.log_alpha = torch.tensor(np.log(alpha),
        #                               dtype=torch.float32,
        #                               requires_grad=True,
        #                               device=device)

    def save(self, save_path):

        output_dir = self.parent_path + save_path
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.actor_model.state_dict(), output_dir)

    def load(self, save_path):

        input_dir = self.parent_path + save_path
        actor_state_dict = torch.load(input_dir)

        self.actor_model.load_state_dict(actor_state_dict)

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()