import time
import torch.multiprocessing as mp
from models.ppo_model import NN_Model
from models.sac_model import NN_SAC_Model, Shared_Optimizer, Replay_Buffer
from utils.utils import *
from environments.intersections import IntersectionsEnv, PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE
from environments.intersections import SAC_AGENT_STATE_SIZE, GLOBAL_SAC_STATE_SIZE
from copy import deepcopy
from workers.ppo_worker import PPOWorker
from workers.sac_worker import SAC_Cycle_Worker
from workers.rule_worker import RuleBasedWorker
from utils.net_scrape import get_intersection_neighborhoods
import traceback


def train_worker(worker_id, shared_NN, data_collector, optimizer, rollout_counter, constants, device,
                 max_neighborhood_size):
    try:
        s_a = get_state_action_size(PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE, max_neighborhood_size,
                                    constants)
        env = IntersectionsEnv(constants, device, worker_id, False, get_net_path(constants))
        local_NN = NN_Model(s_a['s'], s_a['a'], constants['ppo']['hidden_layer_size'],
                            constants['ppo']['actor_layer_size'], constants['ppo']['critic_layer_size'], device).to(
            device)
        worker = PPOWorker(constants, device, env, None, shared_NN, local_NN, optimizer, worker_id)

        while True:
            rollout = rollout_counter.increment_if_below(constants['episode']['num_train_rollouts'])
            if rollout is None:
                print(f"[Worker {worker_id}] All rollouts complete. Exiting.")
                break
            print(f"[Worker {worker_id}] Rollout Progress: {rollout}/{constants['episode']['num_train_rollouts']}")
            try:
                worker.train_rollout(rollout)
            except Exception as e:
                print(f"[Worker {worker_id}] Error in rollout {rollout}: {e}")
                break

        data_collector.gather_control_timer(env.gather_info_time,
                                            env.control_vehicle_time)  #collect the gather_info_time and control_vehicle_time from the env for each worker
        env._close_connection()
    except Exception as e:
        print(f"[Train Worker {worker_id}] Error: {e}")


def train_worker_SAC(worker_id, shared_NN, optimizer, replay_buffer, data_collector, rollout_counter, constants, device, max_neighborhood_size):
    try:
        s_a = get_state_action_size(SAC_AGENT_STATE_SIZE,
                                    GLOBAL_SAC_STATE_SIZE,
                                    ACTION_SIZE,
                                    max_neighborhood_size,
                                    constants)
        env = IntersectionsEnv(constants, device, worker_id, False, get_net_path(constants))
        local_NN = NN_SAC_Model(s_a['s'],
                                s_a['a'],
                                constants['sac']['hidden_layer_size'],
                                constants['sac']['alpha'],
                                device)

        worker = SAC_Cycle_Worker(constants, device, env, optimizer, replay_buffer, None, shared_NN, local_NN, worker_id)

        while True:
            rollout = rollout_counter.increment_if_below(constants['episode']['num_train_rollouts'])
            if rollout is None:
                print(f"[Worker {worker_id}] All rollouts complete. Exiting.")
                break
            print(f"[Worker {worker_id}] Rollout Progress: {rollout}/{constants['episode']['num_train_rollouts']}")
            try:
                worker.train_rollout(rollout)
            except Exception as e:
                print(f"[Worker {worker_id}] Error in rollout {rollout}: {e}")
                break

        data_collector.gather_control_timer(env.gather_info_time,
                                            env.control_vehicle_time)  # collect the gather_info_time and control_vehicle_time from the env for each worker
        env._close_connection()
    except Exception as e:
        print(f"[Train Worker {worker_id}] Error: {e}")


def eval_worker(worker_id, shared_NN, data_collector, rollout_counter, constants, device, max_neighborhood_size):
    print('[===EVAL_WORKING===]')
    try:
        s_a = get_state_action_size(PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE, max_neighborhood_size,
                                    constants)
        env = IntersectionsEnv(constants, device, worker_id, True, get_net_path(constants))
        local_NN = NN_Model(s_a['s'], s_a['a'], constants['ppo']['hidden_layer_size'],
                            constants['ppo']['actor_layer_size'], constants['ppo']['critic_layer_size'], device).to(
            device)
        worker = PPOWorker(constants, device, env, data_collector, shared_NN, local_NN, None, worker_id)
        last_eval = 0

        while True:
            curr_r = rollout_counter.get()
            # print(f"[Eval Worker {id}] curr_r={curr_r}, last_eval={last_eval}")
            if curr_r >= constants['episode']['num_train_rollouts']:
                break
            if curr_r % constants['episode']['eval_freq'] == 0 and last_eval != curr_r:
                last_eval = curr_r
                worker.eval_episodes(curr_r, model_state=worker.NN.state_dict())

        last_eval = curr_r
        worker.eval_episodes(curr_r, model_state=worker.NN.state_dict())
        env._close_connection()
    except Exception as e:
        print(f"[Eval Worker {worker_id}] Error: {e}")
        traceback.print_exc()


def test_worker(worker_id, ep_counter, constants, device, worker=None, data_collector=None, shared_NN=None,
                max_neighborhood_size=None):
    try:
        if not worker:
            s_a = get_state_action_size(SAC_AGENT_STATE_SIZE, GLOBAL_SAC_STATE_SIZE, ACTION_SIZE, max_neighborhood_size,
                                        constants)
            env = IntersectionsEnv(constants, device, worker_id, True, get_net_path(constants))
            local_NN = NN_Model(s_a['s'], s_a['a'], constants['ppo']['hidden_layer_size'],
                                constants['ppo']['actor_layer_size'], constants['ppo']['critic_layer_size'], device).to(
                device)
            worker = PPOWorker(constants, device, env, data_collector, shared_NN, local_NN, None, worker_id)

        while True:
            ep_count = ep_counter.increment_if_below(constants['episode']['test_num_eps'],
                                                     amt=constants['episode']['eval_num_eps'])
            if ep_count is None:
                break
            worker.eval_episodes(None, ep_count=ep_count)

        # print(worker_id, worker.env.gather_info_time)
        data_collector.gather_control_timer(worker.env.gather_info_time,
                                            worker.env.control_vehicle_time)  #collect the gather_info_time and control_vehicle_time from the env for each worker
        # env._close_connection()
        print(f"[Worker {worker.env.agent_ID}] Finished eval_episodes!", flush=True)
    except Exception as e:
        print(f"[Test Worker {worker_id}] Error: {e}")
        traceback.print_exc()


def train_PPO(constants, device, data_collector):
    _, max_neighborhood_size = get_intersection_neighborhoods(get_net_path(constants))
    s_a = get_state_action_size(PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE, max_neighborhood_size, constants)
    shared_NN = NN_Model(s_a['s'], s_a['a'], constants['ppo']['hidden_layer_size'],
                         constants['ppo']['actor_layer_size'], constants['ppo']['critic_layer_size'], device).to(device)
    shared_NN.share_memory()
    optimizer = torch.optim.Adam(shared_NN.parameters(), constants['ppo']['learning_rate'])
    rollout_counter = Counter()
    processes = []

    eval_id = 9999  # reserve high numeric ID for eval
    p = mp.Process(target=eval_worker, args=(eval_id, shared_NN, data_collector, rollout_counter, constants, device, max_neighborhood_size))
    p.start()
    processes.append(p)
    time.sleep(0.5)

    for i in range(constants['parallel']['num_workers']):
        p = mp.Process(target=train_worker, args=(
            i, shared_NN, data_collector, optimizer, rollout_counter, constants, device, max_neighborhood_size))
        p.start()
        processes.append(p)
        time.sleep(0.5)

    for p in processes:
        p.join()


def train_SAC(constants, device, data_collector):
    _, max_neighborhood_size = get_intersection_neighborhoods(get_net_path(constants))
    s_a = get_state_action_size(SAC_AGENT_STATE_SIZE, GLOBAL_SAC_STATE_SIZE, ACTION_SIZE, max_neighborhood_size,
                                constants)

    shared_NN = NN_SAC_Model(s_a['s'],
                             s_a['a'],
                             constants['sac']['hidden_layer_size'],
                             constants['sac']['alpha'],
                             device)
    shared_NN.share_memory()

    optimizer = Shared_Optimizer(shared_NN.actor_model.parameters(),
                                 shared_NN.critic_model.parameters(),
                                 shared_NN.log_alpha,
                                 constants['sac']['learning_rate'],
                                 constants['sac']['weight_decay'])

    replay_buffer = Replay_Buffer(constants['sac']['buffer_length'],
                                  SAC_AGENT_STATE_SIZE + GLOBAL_SAC_STATE_SIZE,
                                  ACTION_SIZE,
                                  constants['sac']['batch_size'],
                                  device)

    rollout_counter = Counter()
    processes = []

    eval_id = 9999  # reserve high numeric ID for eval
    p = mp.Process(target=eval_worker, args=(eval_id, shared_NN, data_collector, rollout_counter, constants, device, max_neighborhood_size))
    p.start()
    processes.append(p)
    time.sleep(0.5)

    for i in range(constants['parallel']['num_workers']):
        p = mp.Process(target=train_worker_SAC,
                       args=(i, shared_NN, optimizer, replay_buffer,
                             data_collector, rollout_counter, constants,
                             device, max_neighborhood_size))
        p.start()
        processes.append(p)
        time.sleep(0.5)

    for p in processes:
        p.join()


def test_PPO(constants, device, data_collector, loaded_model):
    _, max_neighborhood_size = get_intersection_neighborhoods(get_net_path(constants))
    s_a = get_state_action_size(PER_AGENT_STATE_SIZE, GLOBAL_STATE_SIZE, ACTION_SIZE, max_neighborhood_size, constants)
    shared_NN = NN_Model(s_a['s'], s_a['a'], constants['ppo']['hidden_layer_size'],
                         constants['ppo']['actor_layer_size'], constants['ppo']['critic_layer_size'], device).to(device)
    shared_NN.load_state_dict(loaded_model)
    shared_NN.share_memory()
    ep_counter = Counter()
    processes = []

    for i in range(constants['parallel']['num_workers']):
        p = mp.Process(target=test_worker,
                       args=(i, ep_counter, constants, device, None, data_collector, shared_NN, max_neighborhood_size))
        p.start()
        processes.append(p)
        time.sleep(0.5)

    for p in processes:
        p.join()


def test_rule_based(constants, device, data_collector):
    _, max_neighborhood_size = get_intersection_neighborhoods(get_net_path(constants))
    rule_set_class = get_rule_set_class(constants['rule']['rule_set'])
    ep_counter = Counter()
    processes = []

    for i in range(constants['parallel']['num_workers']):
        env = IntersectionsEnv(constants, device, i, True, get_net_path(constants))
        rule_set_params = deepcopy(constants['rule']['rule_set_params'])
        rule_set_params['phases'] = env.phases
        worker = RuleBasedWorker(constants, device, env,
                                 rule_set_class(rule_set_params, get_net_path(constants), constants), data_collector, i)
        p = mp.Process(target=test_worker,
                       args=(i, ep_counter, constants, device, worker, data_collector, None, max_neighborhood_size))
        p.start()
        processes.append(p)
        time.sleep(0.5)

    for p in processes:
        p.join()
