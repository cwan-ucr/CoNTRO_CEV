import time
import torch.multiprocessing as mp
from utils.utils import *
from utils.random_search import grid_choices_random
from utils.grid_search import grid_choices, get_num_grid_choices
from run_agent_parallel import train_PPO, train_SAC, test_rule_based, test_PPO
from run_agent_centralized import train_centralized_PPO
from run_agent_independent import train_ippo
import sys
import os
from data_collector import DataCollector, POSSIBLE_DATA
from utils.tls_config_checker import TLSConfigChecker
from environments.intersections import get_env_name


def run_grid_search(verbose, num_repeat_experiment, df_path=None, overwrite=True, data_to_collect=POSSIBLE_DATA,
                    MVP_key='waitingTime', save_model=True):
    grid = load_constants('constants/constants-grid.json')

    # Make the grid choice generator
    bases, num_choices = get_num_grid_choices(grid)
    grid_choice_gen = grid_choices(grid, bases)
    for diff_experiment, constants in enumerate(grid_choice_gen):
        data_collector_obj = DataCollector(data_to_collect, MVP_key, constants,
                                           'test' if constants['agent']['agent_type'] == 'rule' else 'eval',
                                           df_path, overwrite if diff_experiment == 0 else False, verbose)

        for same_experiment in range(num_repeat_experiment):
            print(' --- Running experiment {}.{} / {}.{} --- '.format(diff_experiment+1, same_experiment+1,
                                                                      num_choices, num_repeat_experiment))
            if save_model: data_collector_obj.set_save_model_path(
                'models/saved_models/grid_{}-{}.pt'.format(diff_experiment + 1, same_experiment + 1))
            run_experiment(diff_experiment+1, same_experiment+1, constants, data_collector_obj)


def run_random_search(verbose, num_diff_experiments, num_repeat_experiment, allow_duplicates=False, df_path=None,
                      overwrite=True, data_to_collect=POSSIBLE_DATA, MVP_key='waitingTime', save_model=True):
    grid = load_constants('constants/constants-grid.json')

    if not allow_duplicates:
        _, num_choices = get_num_grid_choices(grid)
        num_diff_experiments = min(num_choices, num_diff_experiments)
    # Make grid choice generator
    grid_choice_gen = grid_choices_random(grid, num_diff_experiments)
    for diff_experiment, constants in enumerate(grid_choice_gen):
        data_collector_obj = DataCollector(data_to_collect, MVP_key, constants,
                                           'test' if constants['agent']['agent_type'] == 'rule' else 'eval',
                                           df_path, overwrite if diff_experiment == 0 else False, verbose)

        for same_experiment in range(num_repeat_experiment):
            print(' --- Running experiment {}.{} / {}.{} --- '.format(diff_experiment+1, same_experiment+1,
                                                                      num_diff_experiments, num_repeat_experiment))
            if save_model: data_collector_obj.set_save_model_path('models/saved_models/random_{}-{}.pt'.
                                                                  format(diff_experiment+1, same_experiment+1))
            run_experiment(diff_experiment+1, same_experiment+1, constants, data_collector_obj)


def run_normal(verbose, num_experiments=1, df_path=None, overwrite=True, data_to_collect=POSSIBLE_DATA,
               MVP_key='waitingTime', save_model=True, load_model_file=None):
    # if loading, then dont save
    if load_model_file:
        save_model = False

    if not df_path:
        df_path = 'run-data.xlsx'  # def. path

    # Load constants
    constants = load_constants('constants/constants.json')
    # Check tls static or actuated
    net_file = f"data/{get_env_name(constants)}.net.xml"
    checker = TLSConfigChecker(net_file)
    checker.detect_actuated_tls()
    constants['environment']['actuated_tls'] = checker.is_actuated_mode()
    print("Actuated mode detected:", constants['environment']['actuated_tls'])
    # Gather vehicle info or not
    if not constants['environment']['gather_vehicle_info']:
        data_to_collect = data_to_collect[:3]
    data_collector_obj = DataCollector(data_to_collect, MVP_key, constants,
                                       'test' if constants['agent']['agent_type'] == 'rule' or load_model_file else 'eval',
                                       df_path, overwrite, verbose)

    loaded_model = None
    if load_model_file:
        loaded_model = torch.load('models/saved_models/' + load_model_file)

    for exp in range(num_experiments):
        print(' --- Running experiment {} / {} --- '.format(exp + 1, num_experiments))
        if save_model: data_collector_obj.set_save_model_path('models/saved_models/normal_{}.pt'.format(exp+1))
        run_experiment(exp+1, None, constants, data_collector_obj, loaded_model=loaded_model)


def run_experiment(exp1, exp2, constants, data_collector_obj, loaded_model=None):
    data_collector_obj.start_timer()

    if constants['agent']['centralized_train']:
        if loaded_model:
            # test_centralized_PPO(constants, device, data_collector_obj, loaded_model)
            pass #test_centralized_PPO is not developed
        if constants['agent']['agent_type'] == 'ppo':
            train_centralized_PPO(constants, device, data_collector_obj)
    elif constants['agent']['independent_train']:
        if loaded_model:
            # test_ippo(constants, device, data_collector_obj, loaded_model)
            pass #test_ippo is not developed
        if constants['agent']['agent_type'] == 'ppo':
            train_ippo(constants, device, data_collector_obj)
    else:
        if loaded_model:
            test_PPO(constants, device, data_collector_obj, loaded_model)
        elif constants['agent']['agent_type'] == 'ppo':
            train_PPO(constants, device, data_collector_obj)
        elif constants['agent']['agent_type'] == 'sac':
            train_SAC(constants, device, data_collector_obj)
        else:
            assert constants['agent']['agent_type'] == 'rule'
            test_rule_based(constants, device, data_collector_obj)

    # Save and Refresh the data_collector
    data_collector_obj.end_timer(printIt=True)
    data_collector_obj.process_data()
    data_collector_obj.print_summary(exp1, exp2)
    data_collector_obj.done_with_experiment()


if __name__ == '__main__':
    # we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    #GPU
    mp.set_start_method('spawn', force=True)
    # device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    #CPU
    # device = torch.device('cpu')
    
    # df_path = 'run-data.xlsx'

    # print('Num cores: {}'.format(mp.cpu_count()))

    # PPO TSC train
    run_normal(verbose=False, num_experiments=1, df_path='4x4-run-data-rule-ppo-TSC-train-phase_stability_in_reward-limited_cycle_120-more_layers.xlsx', overwrite=True,
                data_to_collect=POSSIBLE_DATA, MVP_key='waitingTime', save_model=True)
    # PPO TSC test
    # run_normal(verbose=False, num_experiments=1, df_path='4*4-run-data-rule-ppo-TSC-test-cycle_120.xlsx', overwrite=True,
    #            data_to_collect=POSSIBLE_DATA, MVP_key='waitingTime', save_model=True, load_model_file='4*4_normal_1_new_reward_limited_cycle_120.pt')
    # rule-based TSC test
    # run_normal(verbose=False, num_experiments=1, df_path='4*4-run-data-rule-rule-TSC-test_phase_stability_in_reward-cycle_120.xlsx', overwrite=True,
    #            data_to_collect=POSSIBLE_DATA, MVP_key='waitingTime', save_model=True)

    # run_random_search(verbose=False, num_diff_experiments=800, num_repeat_experiment=3, allow_duplicates=False,
    #                   df_path='run-data.xlsx', overwrite=True, data_to_collect=POSSIBLE_DATA, MVP_key='waitingTime',
    #                   save_model=True)

    # run_grid_search(verbose=False, num_repeat_experiment=3, df_path='run-data.xlsx', overwrite=True,
    #                 data_to_collect=POSSIBLE_DATA, MVP_key='waitingTime', save_model=True)

