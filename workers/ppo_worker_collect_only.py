import time
import numpy as np
import torch
import torch.nn as nn
from utils.utils import Storage, tensor, random_sample, ensure_shared_grads
from workers.worker import Worker
from torch.cuda.amp import GradScaler
import os, datetime
from multiprocessing import Manager

class PPOWorkerCO(Worker):
    def __init__(self, constants, device, model, env, id, data_collector):
        super(PPOWorkerCO, self).__init__(constants, device, env, id, data_collector)
        self.NN = model
        self.device = device
        self.constants = constants
        self.state = self.env.reset()
        self.ep_step = 0
        self.num_agents = len(env.intersections) if not self.constants['agent']['single_agent'] else 1

    def _stack(self, val):
        return np.stack([val] * self.num_agents)
    
    def _reset(self):
        pass
    
    def _get_prediction(self, states, actions=None, ep_step=None):
        return self.NN(tensor(states, self.device), actions)

    def _get_action(self, prediction):
        return prediction['a'].cpu().numpy()

    
    def _copy_shared_model_to_local(self):
        pass

    def collect_rollout(self, model_state):
        # Load model
        self.NN.load_state_dict(model_state)
        self.NN.to(self.device)
        self.NN.eval()

        storage = Storage(self.constants['episode']['rollout_length'])
        state = np.copy(self.state)
        rollout_amt = 0

        while rollout_amt < self.constants['episode']['rollout_length']:
            # print(f"[Worker {self.id}] Rollout step {rollout_amt}/{self.constants['episode']['rollout_length']}")
            with torch.no_grad():
                prediction = self.NN(tensor(state, self.device))
            action = prediction['a'].cpu().numpy()
            next_state, reward, done, _ = self.env.step(action, self.ep_step, get_global_reward=False)

            self.ep_step += 1
            if done:
                self.ep_step = 0
                self.state = self.env.reset()

            if self.ep_step > self.constants['episode']['warmup_ep_steps']:
                storage.add(prediction)
                storage.add({
                    'r': tensor(reward, self.device).unsqueeze(-1),
                    'm': tensor(self._stack(1 - done), self.device).unsqueeze(-1),
                    's': tensor(state, self.device)
                })
                rollout_amt += 1

            state = np.copy(next_state)

        self.state = np.copy(state)
        prediction = self.NN(tensor(state, self.device))
        storage.add(prediction)
        storage.placeholder()

        # === Compute GAE returns and advantages ===
        advantages = torch.zeros((self.num_agents, 1), device=self.device)
        returns = prediction['v'].detach()

        for i in reversed(range(self.constants['episode']['rollout_length'])):
            reward = storage.r[i]
            mask = storage.m[i]
            value = storage.v[i]
            next_value = storage.v[i + 1]

            returns = reward + self.constants['ppo']['discount'] * mask * returns
            td_error = reward + self.constants['ppo']['discount'] * mask * next_value - value
            advantages = td_error + self.constants['ppo']['discount'] * self.constants['ppo']['gae_tau'] * mask * advantages

            storage.ret[i] = returns.detach()
            storage.adv[i] = advantages.detach()

        storage.to_cpu_and_detach()
        return storage
    