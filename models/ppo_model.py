import torch.nn as nn
import torch
import numpy as np
from utils.utils import plot_grad_flow, layer_init_filter


# Simple one layer
class ModelBody(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(ModelBody, self).__init__()
        self.name = 'model_body'

        layers = []
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size

        self.model = nn.Sequential(*layers)
        self.model.apply(layer_init_filter)

    def forward(self, states):
        return self.model(states)


class ActorModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, action_size):
        super(ActorModel, self).__init__()
        self.name = 'actor'
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, action_size))
        self.model = nn.Sequential(*layers)
        self.model.apply(layer_init_filter)

    def forward(self, hidden_states):
        return self.model(hidden_states)


class CriticModel(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(CriticModel, self).__init__()
        self.name = 'critic'
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)
        self.model.apply(layer_init_filter)

    def forward(self, hidden_states):
        return self.model(hidden_states)


class NN_Model(nn.Module):
    def __init__(self, state_size, action_size, body_hidden_sizes, actor_hidden_sizes, critic_hidden_sizes, device):
        super(NN_Model, self).__init__()
        self.body_model = ModelBody(state_size, body_hidden_sizes).to(device)
        body_output_size = body_hidden_sizes[-1]
        self.actor_model = ActorModel(body_output_size, actor_hidden_sizes, action_size).to(device)
        self.critic_model = CriticModel(body_output_size, critic_hidden_sizes).to(device)
        self.models = [self.body_model, self.actor_model, self.critic_model]

    def forward(self, states, actions=None):
        hidden_states = self.body_model(states)
        v = self.critic_model(hidden_states)
        logits = self.actor_model(hidden_states)
        dist = torch.distributions.Categorical(logits=logits)
        if actions is None:
            actions = dist.sample()
        log_prob = dist.log_prob(actions).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return {
            'a': actions,
            'log_pi_a': log_prob,
            'ent': entropy,
            'v': v
        }
