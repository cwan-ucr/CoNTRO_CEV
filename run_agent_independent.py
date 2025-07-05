# === Refactored version for IPPO with multi-threaded training ===

import time
import torch
import multiprocessing as mp
from models.ppo_model import NN_Model
from utils.utils import *
from environments.intersections import IntersectionsEnv, PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE
from workers.ippo_worker_collect_only import IPPOWorkerCO
import copy
from utils.net_scrape import get_intersection_neighborhoods
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def rollout_process(worker_id, constants, device, model_states_dicts, env_class, net_path, rollout_results, data_collector):
    env = env_class(constants, device, agent_ID=worker_id, eval_agent=False, net_path=net_path)
    worker = IPPOWorkerCO(constants, device, model_states_dicts, env, worker_id, data_collector)
    start = time.time()
    storage_per_agent = worker.collect_rollout()
    print(f"[Worker {worker_id}] Finished rollout in {time.time() - start:.2f} sec")
    rollout_results.append(storage_per_agent)

def train_one_agent(agent_id, constants, device, shared_model, optimizer, storage):
    states, actions, log_probs_old, returns, advantages = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(constants['ppo']['optimization_epochs']):
        sampler = random_sample(np.arange(states.size(0)), constants['ppo']['minibatch_size'])
        for batch_indices in sampler:
            batch_indices = torch.tensor(batch_indices).long()
            sampled_states = states[batch_indices].to(device)
            sampled_actions = actions[batch_indices].to(device)
            sampled_log_probs_old = log_probs_old[batch_indices].to(device)
            sampled_returns = returns[batch_indices].to(device)
            sampled_advantages = advantages[batch_indices].to(device)

            prediction = shared_model(sampled_states, sampled_actions)
            ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
            obj = ratio * sampled_advantages
            obj_clipped = ratio.clamp(
                1.0 - constants['ppo']['ppo_ratio_clip'],
                1.0 + constants['ppo']['ppo_ratio_clip']
            ) * sampled_advantages

            policy_loss = -torch.min(obj, obj_clipped).mean() - \
                          constants['ppo']['entropy_weight'] * prediction['ent'].mean()
            value_loss = constants['ppo']['value_loss_coef'] * \
                         (sampled_returns - prediction['v']).pow(2).mean()
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            if constants['ppo']['clip_grads']:
                torch.nn.utils.clip_grad_norm_(shared_model.parameters(), constants['ppo']['gradient_clip'])
            optimizer.step()
    shared_model.eval()
    return agent_id

def parallel_rollout_and_train_ippo(constants, device, env_class, shared_models, optimizers, data_collector):
    model_states_dicts = [
        {k: v.cpu() for k, v in shared_models[i].state_dict().items()} for i in range(len(shared_models))
    ]
    net_path = get_net_path(constants)

    manager = mp.Manager()
    rollout_results = manager.list()
    processes = []

    for i in range(constants['parallel']['num_workers']):
        p = mp.Process(
            target=rollout_process,
            args=(i, constants, device, model_states_dicts, env_class, net_path, rollout_results, data_collector)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    num_agents = len(shared_models)
    storages = [Storage(constants['episode']['rollout_length']) for _ in range(num_agents)]

    for rollout_per_worker in rollout_results:
        for i in range(num_agents):
            storages[i].merge(rollout_per_worker[i])

    with ThreadPoolExecutor(max_workers=num_agents) as executor:
        futures = [executor.submit(train_one_agent, i, constants, device, shared_models[i], optimizers[i], storages[i]) for i in range(num_agents)]
        for f in futures:
            f.result()

    print("[Training] IPPO models updated after all rollouts.")

def train_ippo(constants, device, data_collector):
    net_path=get_net_path(constants)
    _, max_neighborhood_size = get_intersection_neighborhoods(net_path)
    s_a = get_state_action_size(PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE, max_neighborhood_size, constants)
    
    eval_env = IntersectionsEnv(constants, device, agent_ID=9999, eval_agent=True, net_path=net_path)
    
    num_agents = len(eval_env.intersections)

    shared_models = [NN_Model(s_a['s'], s_a['a'], constants['ppo']['hidden_layer_size'], constants['ppo']['actor_layer_size'], constants['ppo']['critic_layer_size'], device).to(device)
                     for _ in range(num_agents)]
    optimizers = [torch.optim.Adam(shared_models[i].parameters(), lr=constants['ppo']['learning_rate']) for i in range(num_agents)]

    

    for rollout_id in range(constants['episode']['num_train_rollouts']):
        print(f"[IPPO Rollout {rollout_id}/{constants['episode']['num_train_rollouts']}]")
        parallel_rollout_and_train_ippo(constants, device, IntersectionsEnv, shared_models, optimizers, data_collector)

    eval_env._close_connection()
