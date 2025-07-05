import time
import torch
import multiprocessing as mp
from models.ppo_model import NN_Model
from utils.utils import *
from environments.intersections import IntersectionsEnv, PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE
from workers.ppo_worker_collect_only import PPOWorkerCO
import copy
from utils.net_scrape import get_intersection_neighborhoods
import numpy as np

def rollout_process(worker_id, constants, device, model, env_class, net_path, model_state_dict_cpu, rollout_results, data_collector):
    env = env_class(constants, device, agent_ID=worker_id, eval_agent=False, net_path=net_path)
    worker = PPOWorkerCO(constants, device, model, env, worker_id, data_collector)
    worker.NN.load_state_dict(model_state_dict_cpu)
    worker.NN.to(device)

    start = time.time()
    storage = worker.collect_rollout(model_state_dict_cpu)
    print(f"[Worker {worker_id}] Finished rollout in {time.time() - start:.2f} sec")
    rollout_results.append(storage)

def parallel_rollout_and_train(constants, device, env_class, shared_model, optimizer, data_collector):
    model_state_dict_cpu = {k: v.cpu() for k, v in shared_model.state_dict().items()}
    net_path = get_net_path(constants)

    manager = mp.Manager()
    rollout_results = manager.list()
    processes = []

    for i in range(constants['parallel']['num_workers']):
        p = mp.Process(
            target=rollout_process,
            args=(i, constants, device, shared_model, env_class, net_path, model_state_dict_cpu, rollout_results, data_collector)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    shared_model.to(device)
    shared_model.train()

    if len(rollout_results) == 0:
        print("[Warning] No rollout results were collected. Skipping training.")
        return

    all_storage = combine_storage(list(rollout_results))
    states, actions, log_probs_old, returns, advantages = all_storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(constants['ppo']['optimization_epochs']):
        sampler = random_sample(np.arange(states.size(0)), constants['ppo']['minibatch_size'])
        for batch_indices in sampler:
            batch_indices = torch.tensor(batch_indices).long()  # Keep it on CPU since states are on CPU
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
    print("[Training] Shared model updated after all rollouts.")

def train_centralized_PPO(constants, device, data_collector):
    _, max_neighborhood_size = get_intersection_neighborhoods(get_net_path(constants))
    s_a = get_state_action_size(PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE, max_neighborhood_size, constants)

    model = NN_Model(s_a['s'], s_a['a'], constants['ppo']['hidden_layer_size'], constants['ppo']['actor_layer_size'], constants['ppo']['critic_layer_size'], device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=constants['ppo']['learning_rate'])
    
    eval_env = IntersectionsEnv(constants, device, agent_ID=9999, eval_agent=True, net_path=get_net_path(constants))
    eval_worker = PPOWorkerCO(constants, device, model, eval_env, id=9999, data_collector=data_collector)

    for rollout_id in range(constants['episode']['num_train_rollouts']):
        print(f"[Centralized Rollout {rollout_id}/{constants['episode']['num_train_rollouts']}]")
        parallel_rollout_and_train(constants, device, IntersectionsEnv, model, optimizer, data_collector)

        if (rollout_id + 1) % constants['episode']['eval_freq'] == 0:
            print(f"[Eval at Rollout {rollout_id + 1}]")
            model.eval()
            eval_reward = eval_worker.eval_episodes(rollout_id + 1, model_state=copy.deepcopy(model.state_dict()))
            if eval_reward is not None:
                print(f"[Eval Result] Avg Reward: {eval_reward:.3f}")

    data_collector.gather_control_timer(eval_env.gather_info_time, eval_env.control_vehicle_time)
    eval_env._close_connection()
