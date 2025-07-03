import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import Storage_SAC, tensor, random_sample, ensure_shared_grads
from workers.worker import Worker


# Code adapted from: Shangtong Zhang (https://github.com/ShangtongZhang)
class SAC_Cycle_Worker(Worker):
    def __init__(self, constants, device, env, optimizer, replay_buffer, data_collector, shared_NN, local_NN, id, dont_reset=False):
        super(SAC_Cycle_Worker, self).__init__(constants, device, env, id, data_collector)
        self.NN = local_NN
        self.shared_NN = shared_NN
        if not dont_reset:  # for the vis agent script this messes things up
            state = self.env.SAC_reset()
            state = [self.env.signal_state_encoder(np.array(state[i]), i) for i in range(len(state))]
            self.env.initialize_signal(state)
        self.ep_step = 0
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.num_agents = len(env.intersections) if not self.constants['agent']['single_agent'] else 1
        self.ep_reward = 0
        self.NN.eval()

    def _reset(self):
        pass

    def _get_prediction(self, states, actions=None, ep_step=None):
        return self.NN.actor_model(tensor(states, self.device), actions)

    def _get_action(self, prediction):
        return prediction['a'].cpu().numpy()

    def _take_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(dtype=torch.float32, device=self.device)
            action, _ = self.NN.actor_model(state)
            action = action.to(dtype=torch.float32).cpu().detach().numpy()

        return action.astype(np.float32)

    def _copy_shared_model_to_local(self):
        self.NN.load_state_dict(self.shared_NN.state_dict())

    def _stack(self, val):
        assert not isinstance(val, list)
        return np.stack([val] * self.num_agents)

    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

    def update(self, gama, grad_clip, tau):
        if self.replay_buffer.len() < self.replay_buffer.batch_size:
            return

        # Sampling from replay buffer
        batch_s, batch_a, batch_r, batch_s_, batch_d = self.replay_buffer.sample()

        # ----------------------------- ↓↓↓↓↓ Update QValue Net ↓↓↓↓↓ ------------------------------#
        batch_a_, log_prob_ = self.NN.actor_model(batch_s_)
        entropy = -log_prob_

        Q1_, Q2_ = self.NN.target_critic_model(batch_s_, batch_a_)
        Q_target = (batch_r + gama * (~batch_d) *
                              (torch.min(Q1_.squeeze(1), Q2_.squeeze(1)) + self.NN.log_alpha.exp() * entropy))

        Q1, Q2 = self.NN.critic_model(batch_s, batch_a)

        td_error = F.mse_loss(Q_target, Q1.squeeze(1)) + F.mse_loss(Q_target, Q2.squeeze(1))

        self.optimizer.critic_loss.append(td_error.item())

        self.optimizer.optim_critic.zero_grad()
        td_error.backward()
        torch.nn.utils.clip_grad_norm_(self.NN.critic_model.parameters(), grad_clip)
        ensure_shared_grads(self.NN.critic_model, self.shared_NN.critic_model)
        self.optimizer.optim_critic.step()

        # ----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        for params in self.NN.critic_model.parameters(): params.requires_grad = False
        new_actions, new_log_prob = self.NN.actor_model(batch_s)

        Q1, Q2 = self.NN.critic_model(batch_s, batch_a)
        min_q = torch.min(Q1, Q2)

        actor_loss = (self.NN.log_alpha.exp() * new_log_prob - min_q.squeeze(1)).mean()
        self.optimizer.actor_loss.append(actor_loss.item())

        self.optimizer.optim_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.NN.actor_model.parameters(), grad_clip)
        ensure_shared_grads(self.NN.actor_model, self.shared_NN.actor_model)
        self.optimizer.optim_actor.step()

        for params in self.NN.critic_model.parameters(): params.requires_grad = True

        # ----------------------------- ↓↓↓↓↓ Update Alpha ↓↓↓↓↓ ------------------------------#
        alpha_loss = -((self.NN.log_alpha * new_log_prob + self.NN.target_entropy).detach()).mean()
        alpha_loss.requires_grad = True
        self.optimizer.alpha_loss.append(alpha_loss.item())

        self.optimizer.optim_alpha.zero_grad()
        alpha_loss.backward()
        self.shared_NN.log_alpha.grad = self.NN.log_alpha.grad
        self.optimizer.optim_alpha.step()

        # soft update
        self.soft_update(self.shared_NN.critic_model, self.shared_NN.target_critic_model, tau)

    def train_rollout(self, total_step):

        print(f"[Worker {self.env.agent_ID}] Entered train_rollout() with rollout={total_step}")
        done = False

        # Sync.
        self._copy_shared_model_to_local()
        rollout_amt = 0  # keeps track of the amt in the current rollout storage
        # self.env.vis = True if total_step % 1 == 0 else False
        while rollout_amt < self.constants['episode']['rollout_length']:
            cycle_done = self.env.get_cycle_done()  # check if the signal cycle for each intersection is done
            step_state = self.env.get_step_state()

            for i, intersection in enumerate(self.env.intersections):
                self.env.cycle_state_origin[intersection].append(step_state[i])
                if cycle_done[i]:
                    reward_i = self.env.get_cycle_reward(intersection)
                    self.ep_reward += reward_i
                    original_state = self.env.cycle_state_origin[intersection]
                    cycle_state = self.env.signal_state_encoder(np.array(original_state), i)

                    action_i = self._take_action(cycle_state)
                    self.env.prev_cycle_action[intersection] = self.env.cycle_action[intersection] if self.env.cycle_action[intersection] is not None else None
                    self.env._excute_cycle_action(action_i, intersection, is_start=False)

                    self.env.cycle_state[intersection] = cycle_state.copy()
                    self.env.cycle_action[intersection] = action_i.copy()

                    if self.env.prev_cycle_action[intersection] is not None:
                        self.replay_buffer.add(self.env.prev_cycle_state[intersection].copy(),
                                               self.env.prev_cycle_action[intersection].copy(),
                                               reward_i,
                                               self.env.cycle_state[intersection].copy(),
                                               done)
                        self.update(self.constants['sac']['gama'], self.constants['sac']['grad_clip'], self.constants['sac']['tau'])

            rollout_amt += 1



            done = self.env.vehicle_step(self.ep_step)  # vehicle controller
            self.ep_step += 1
            if done:
                # Sync local model with shared model at start of each ep
                print(f"[Worker {self.env.agent_ID}] Rollout reward stats {self.ep_reward:.6f}")
                self._copy_shared_model_to_local()
                self.ep_step = 0
                self.ep_reward = 0
                break

        return total_step
