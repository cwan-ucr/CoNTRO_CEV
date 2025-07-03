from multiprocessing import Manager, Lock, Value
import pandas as pd
import os.path
from os import path
from collections import OrderedDict
import time
import torch
from copy import deepcopy
import numpy as np


# What data is possible to be collected
POSSIBLE_DATA = ['time',
                 'rew',
                 'waitingTime',
                 'avg_speed_car',
                 'energy_per_mile_car',
                 'avg_veh_num_car']

# Keys that must be added
_EVAL_DEF_KEYS = ['rollout']
_TEST_DEF_KEYS = []

_GLOBAL_DATA = {'time': {'header': 'total time taken(m)', 'init_value': 0, 'add_type': 'set'}}
_EVAL_DATA = {'rew': {'header': 'eval_best_rew', 'init_value': float('-inf'), 'add_type': 'set_on_greater'},
              'waitingTime': {'header': 'eval_best_waiting_time', 'init_value': float('inf'), 'add_type': 'set_on_less'},
              'rollout': {'header': 'eval_train_rollout', 'init_value': float('inf'), 'add_type': 'set_on_less'}}
_EVAL_DATA.update({
    'avg_speed_car': {'header': 'avg_speed_ev', 'init_value': float('-inf'), 'add_type': 'set_on_greater'},
    'energy_per_mile_car': {'header': 'energy_per_mile_ev', 'init_value': float('inf'), 'add_type': 'set_on_less'},
    'avg_veh_num_car': {'header': 'avg_veh_num_car', 'init_value': float('inf'), 'add_type': 'set_on_less'}})
_TEST_DATA = {'rew': {'header': 'test_avg_rew', 'init_value': [], 'add_type': 'append'},
            'waitingTime': {'header': 'test_avg_waiting_time', 'init_value': [], 'add_type': 'append'}}
_TEST_DATA.update({
    'avg_speed_car': {'header': 'test_avg_speed_ev', 'init_value': [], 'add_type': 'append'},
    'energy_per_mile_car': {'header': 'test_energy_per_mile_ev', 'init_value': [], 'add_type': 'append'},
    'avg_veh_num_car': {'header': 'test_avg_veh_num_car', 'init_value': [], 'add_type': 'append'}})

class DataCollector:
    # Will make a df or add to an existing df (if a file already exists in the path)
    def __init__(self, data_keys, mvp_key, constants, eval_or_test, df_path, overwrite, verbose):
        assert len(set(data_keys)) == len(data_keys), 'There are duplicate keys in data_keys: {}'.format(data_keys)
        # self.manager = Manager()
        self.data_keys = deepcopy(data_keys)
        self.eval_or_test = eval_or_test
        if self.eval_or_test == 'eval':
            assert mvp_key in self.data_keys, 'Need MVP key in the set of sent in data keys: {} not in {}'.format(mvp_key, self.data_keys)
            if self.data_keys[0] != mvp_key:
                self.data_keys.remove(mvp_key)
                self.data_keys.insert(0, mvp_key)
        self._add_def_keys()  # add def keys
        assert len(set(data_keys)) == len(data_keys), 'There are duplicate keys in data_keys: {}'.format(
            self.data_keys)
        self.mvp_key = mvp_key
        self.constants = constants
        self._make_data_dict()
        self.df_path = df_path
        # If overwite then make a new one no matter if it already exists, other wise check the path
        self.df = self._check_if_df_exists() if not overwrite else self._make_df()
        self.verbose = verbose
        self.save_model_path = None
        # self.lock = Lock()
        self.gather_info_time = Value('d', 0.0)  # 'd' = double
        self.control_vehicle_time = Value('d', 0.0)

    def set_save_model_path(self, save_model_path):
        self.save_model_path = save_model_path

    # Check to make sure def keys are in the new_data
    def _check_def_keys(self, new_data):
        li = _EVAL_DEF_KEYS if self.eval_or_test == 'eval' else _TEST_DEF_KEYS
        for k in li:
            assert k in new_data, 'Didnt return the def key {} with new data for collection'.format(k)

    # This is for collecting info at the end of an ep (or chunk of episodes), check MVP value if eval to see if I should
    # add this set of data, because during eval mode I only save data from eps where the MVP value is better than before
    def collect_ep(self, new_data, model_state=None, ep_count=None):
        self._check_def_keys(new_data)
        new_best = False
        set_values = False
        if self.eval_or_test == 'eval':
            assert self.mvp_key in new_data, 'MVP key must be in data for episode collection.'
            if self.info_dict[self.mvp_key]['add_type'] == 'set_on_greater' and new_data[self.mvp_key] > self.data[self.mvp_key]:
                new_best = True
                set_values = True
            elif self.info_dict[self.mvp_key]['add_type'] == 'set_on_less' and new_data[self.mvp_key] < self.data[self.mvp_key]:
                new_best = True
                set_values = True
            # Save if new_best
            if new_best and model_state:
                self._save_model(model_state)
        else:
            set_values = True

        if set_values:
            for key in new_data: self._add_value(key, new_data[key])

        self._print(new_data, new_best, ep_count)
        val = self.data[self.mvp_key]
        count = len(val) if hasattr(val, '__len__') else 1
        print(f"[DataCollector] Recorded eval result with {self.mvp_key}={val} | new_best={new_best} | total_count={count}")


    # Whenever (only if eval) a new best is reached
    def _save_model(self, model_state):
        if self.save_model_path:
            torch.save(model_state, self.save_model_path)

    def _print(self, new_data, new_best, ep_count):
        if not self.verbose: return
        prepend = ''
        if self.eval_or_test == 'eval':
            prepend = 'NEW BEST ' if new_best else ''
            prepend += 'on rollout: {}  '.format(new_data['rollout'])
        if ep_count is not None:
            prepend = 'Current test episode: {}  '.format(ep_count)
        all_print = prepend
        for k in list(self.data.keys()):  # So its in the correct order
            if k in new_data and k != 'rollout':
                all_print += '{}: {:.2f}  '.format(k, new_data[k])
        print(all_print)

    def print_summary(self, exp1, exp2=None):
        if exp2 == None:
            all_print = 'Experiment {} Summary:  '.format(exp1)
        else:
            all_print = 'Experiment {}.{} Summary:  '.format(exp1, exp2)
        # print("=== Debug: Raw data before summary ===")
        for k, v in self.data.items():
            # if isinstance(v, float):
            #     print(f"{k}: {v} (NaN? {np.isnan(v)}, Inf? {np.isinf(v)})")
            # else:
            #     print(f"{k}: {v}")
            if k not in _EVAL_DEF_KEYS + _TEST_DEF_KEYS + POSSIBLE_DATA:
                continue
            if isinstance(v, float):
                all_print += '{}: {:.2f}  '.format(k, v)
            else:
                all_print += '{}: {}  '.format(k, v)
        print(all_print)

    def _add_value(self, k, v):
        if k not in self.data: return
        add_type = self.info_dict[k]['add_type']
        # with self.lock:
        if add_type == 'append':
            # print()
            self.data[k].append(v)
        elif add_type == 'set_on_greater' or add_type == 'set_on_less': self.data[k] = v
        else: assert 1 == 0, add_type

        # print(self.data[k])

    def _add_def_keys(self):
        li = _EVAL_DEF_KEYS if self.eval_or_test == 'eval' else _TEST_DEF_KEYS
        for k in li[::-1]:
            self.data_keys.insert(0, k)

    def _check_if_df_exists(self):
        # If it eists then return it
        if path.exists(self.df_path):
            return pd.read_excel(self.df_path)
        else:
            return self._make_df()

    def _make_df(self):
        df = pd.DataFrame(columns=list(self.data.keys()))
        df.to_excel(self.df_path, index=False, header=list(self.data.keys()))
        return df

    def _add_hyp_param_dict(self):
        def add_hyp_param_dict(append_letter, dic):
            for k, v in list(dic.items()):
                self.data[append_letter + '_' + k] = v
        add_hyp_param_dict('Ep', self.constants['episode'])
        add_hyp_param_dict('A', self.constants['agent'])
        add_hyp_param_dict('P', self.constants['ppo'])
        add_hyp_param_dict('R', self.constants['rule'])
        add_hyp_param_dict('En', self.constants['environment'])
        add_hyp_param_dict('M', self.constants['multiagent'])
        add_hyp_param_dict('P', self.constants['parallel'])
        return self.data

    def _append_to_df(self):
        assert self.df is not None, 'At this point df should not be none.'
        dat = {k: v for k, v in list(self.data.items())}
        self.df = self.df.append(dat, ignore_index=True)

    def _write_to_excel(self):
        self.df.to_excel(self.df_path, index=False)

    # Because one data collector is made for all runs (ie a new one is NOT made per experiment) need to refresh
    def _refresh_data_store(self):
        self._make_data_dict()

    # This is just for getting the mean for array objects in the data dict
    def process_data(self):
        for k, v in list(self.data.items()):
            # print(f"[Debug] Key: {k}, Values: {v}")
            if k in self.info_dict and self.info_dict[k]['add_type'] == 'append':
                self.data[k] = sum(v) / len(v)

    # Signals to this object that it can add to df and and refresh the data store (call when threads are done)
    def done_with_experiment(self):
        self._append_to_df()
        self._write_to_excel()
        self._refresh_data_store()

    def _make_data_dict(self):
        self.info_dict = _GLOBAL_DATA.copy()
        manager = Manager()
        self.data = manager.dict()
        # Add values to collect
        for key in self.data_keys:
            # Check eval or test first then global then error
            if self.eval_or_test == 'eval' and key in _EVAL_DATA:
                self.data[key] = _EVAL_DATA[key]['init_value']
                self.info_dict[key] = _EVAL_DATA[key].copy()
            elif self.eval_or_test == 'test' and key in _TEST_DATA:
                self.data[key] = _TEST_DATA[key]['init_value']
                self.info_dict[key] = _TEST_DATA[key].copy()
            elif key in _GLOBAL_DATA:
                self.data[key] = _GLOBAL_DATA[key]['init_value']
                self.info_dict[key] = _GLOBAL_DATA[key].copy()
            else:
                raise ValueError('Data collector was sent {} which isnt data that can be collected'.format(key))
            if self.data[key] == []: self.data[key] = manager.list()
        # Add hyp param values
        self._add_hyp_param_dict()
        return self.data, self.info_dict

    def start_timer(self):
        self.start_time = time.time()

    def end_timer(self, printIt):
        end_time = time.time()
        total_time = (end_time - self.start_time) / 60.
        print(self.gather_info_time)
        # print the final gather time and control time, divided by the number of workers to get the average value
        gather_time = float(self.gather_info_time.value)
        control_time = float(self.control_vehicle_time.value)

        gather_time /= 60. * self.constants['parallel']['num_workers']
        control_time /= 60. * self.constants['parallel']['num_workers']
        if printIt:
            print(' - Total Time taken (m): {:.2f} - '.format(total_time))
            print(' - Gather Info Time taken (m): {:.2f} - '.format(gather_time))
            print(' - Vehicle Control Time taken (m): {:.2f} - '.format(control_time))
        if 'time' in self.data:
            self.data['time'] = total_time
            
    def gather_control_timer(self, gather_time, control_time):
        self.gather_info_time.value += gather_time
        self.control_vehicle_time.value += control_time
