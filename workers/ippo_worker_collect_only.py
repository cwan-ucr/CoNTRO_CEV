import time
import numpy as np
import torch
import torch.nn as nn
from utils.utils import Storage, tensor, get_state_action_size, get_net_path
from utils.net_scrape import get_intersection_neighborhoods
from environments.intersections import PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE
from workers.worker import Worker

class IPPOWorkerCO(Worker):
    def __init__(self, constants, device, model_states_dicts, env, id, data_collector):
        super(IPPOWorkerCO, self).__init__(constants, device, env, id, data_collector)
        self.constants = constants
        self.device = device
        self.state = self.env.reset()
        self.ep_step = 0
        self.num_agents = len(env.intersections)
        self.models = [self._load_model(i, model_states_dicts[i]) for i in range(self.num_agents)]

    def _load_model(self, agent_id, state_dict):
        from models.ppo_model import NN_Model  # Local import to avoid circular issues
        net_path=get_net_path(self.constants)
        _, max_neighborhood_size = get_intersection_neighborhoods(net_path)
        s_a = get_state_action_size(PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE, max_neighborhood_size, self.constants)
        model = NN_Model(s_a['s'], s_a['a'], self.constants['ppo']['hidden_layer_size'], self.constants['ppo']['actor_layer_size'], self.constants['ppo']['critic_layer_size'], self.device).to(self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _get_prediction(self, agent_id, state):
        with torch.no_grad():
            return self.models[agent_id](tensor(state, self.device).unsqueeze(0))

    def collect_rollout(self):
        storages = [Storage(self.constants['episode']['rollout_length']) for _ in range(self.num_agents)]
        state = np.copy(self.state)
        rollout_amt = 0

        while rollout_amt < self.constants['episode']['rollout_length']:
            predictions = [self._get_prediction(i, state[i]) for i in range(self.num_agents)]
            actions = [pred['a'].item() for pred in predictions]

            next_state, reward, done, _ = self.env.step(actions, self.ep_step, get_global_reward=False)
            self.ep_step += 1
            if done:
                self.ep_step = 0
                self.state = self.env.reset()

            if self.ep_step > self.constants['episode']['warmup_ep_steps']:
                for i in range(self.num_agents):
                    storages[i].add({
                        's': tensor(state[i], self.device).unsqueeze(0),
                        'a': tensor(actions[i], self.device).unsqueeze(0),
                        'log_pi_a': predictions[i]['log_pi_a'],
                        'v': predictions[i]['v'],
                        'ent': predictions[i]['ent'],
                        'r': tensor([reward[i]], self.device).unsqueeze(-1),
                        'm': tensor([1 - done], self.device).unsqueeze(-1)
                    })
                rollout_amt += 1

            state = np.copy(next_state)

        self.state = np.copy(state)
        predictions = [self._get_prediction(i, state[i]) for i in range(self.num_agents)]
        for i in range(self.num_agents):
            storages[i].add(predictions[i])
            storages[i].placeholder()

        for i in range(self.num_agents):
            advantages = torch.zeros((1, 1), device=self.device)
            returns = predictions[i]['v'].detach()
            for j in reversed(range(self.constants['episode']['rollout_length'])):
                reward = storages[i].r[j]
                mask = storages[i].m[j]
                value = storages[i].v[j]
                next_value = storages[i].v[j + 1]

                returns = reward + self.constants['ppo']['discount'] * mask * returns
                td_error = reward + self.constants['ppo']['discount'] * mask * next_value - value
                advantages = td_error + self.constants['ppo']['discount'] * self.constants['ppo']['gae_tau'] * mask * advantages

                storages[i].ret[j] = returns.detach()
                storages[i].adv[j] = advantages.detach()

            storages[i].to_cpu_and_detach()

        return storages

    