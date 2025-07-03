import traci
import numpy as np
from utils.net_scrape import *
import subprocess
import time
import os
from sumolib import checkBinary

# Base class
class Environment:
    def __init__(self, constants, device, agent_ID, eval_agent, net_path, vis=False):
        self.constants = constants
        self.device = device
        self.agent_ID = agent_ID
        self.eval_agent = eval_agent
        self.net_path = net_path
        self.vis = vis
        self.phases = None
        self.agent_type = constants['agent']['agent_type']
        self.single_agent = constants['agent']['single_agent']
        self.intersections = get_intersections(net_path)
        self.intersections_index = {intersection: i for i, intersection in enumerate(self.intersections)}
        self.neighborhoods, self.max_num_neighbors = get_intersection_neighborhoods(net_path)
        self.connection = None
        self.sumo_proc = None
        self.gather_info_time = 0.0
        self.control_vehicle_time = 0.0

        
    # def _cleanup_old_outputs(self):
    #     edge_data_file = f"data/edgeData_{self.agent_ID}.out.xml"
    #     if os.path.exists(edge_data_file):
    #         os.remove(edge_data_file)

    def _make_state(self):
        if self.agent_type == 'rule': return {}
        if self.single_agent: return []
        else:
            if self.constants['multiagent']['state_interpolation'] == 0:
                return [[] for _ in range(len(self.intersections))]
            else: return {}

    def _add_to_state(self, state, value, key, intersection):
        if self.agent_type == 'rule':
            if intersection:
                if intersection not in state: state[intersection] = {}
                state[intersection][key] = value
            else:
                state[key] = value
        else:
            if self.single_agent:
                if isinstance(value, list):
                    state.extend(value)
                else:
                    state.append(value)
            else:
                if self.constants['multiagent']['state_interpolation'] == 0:
                    if isinstance(value, list):
                        state[self.intersections_index[intersection]].extend(value)
                    else:
                        state[self.intersections_index[intersection]].append(value)
                else:
                    if intersection not in state: state[intersection] = []
                    if isinstance(value, list):
                        state[intersection].extend(value)
                    else:
                        state[intersection].append(value)

    def _process_state(self, state):
        if not self.agent_type == 'rule':
            if self.single_agent:
                return np.expand_dims(np.array(state), axis=0)
            else:
                return np.array(state)
        return state

    def _open_connection(self):
        raise NotImplementedError

    def _close_connection(self):
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception as e:
                print(f"[Worker {self.agent_ID}] Warning during connection close: {e}")
            self.connection = None

        if hasattr(self, 'sumo_proc') and self.sumo_proc is not None:
            try:
                self.sumo_proc.terminate()
                self.sumo_proc.wait()
            except Exception as e:
                print(f"[Worker {self.agent_ID}] Warning during SUMO process termination: {e}")
            self.sumo_proc = None

    def reset(self):
        # self._cleanup_old_outputs()
        # === Reset environment state ===
        self._close_connection()
        self._open_connection()

        # === Reset EAD-related tracking ===
        self.phases_old_all = {}
        self.Qab = []
        self.last_speed = []
        self.EAD_ID_list = []
        self.ead_counts_per_step = []  # List to track EAD vehicle counts per step
        
        # Optionally reset vehicle stats if needed
        self.vehicle_stats = {}
        
        # reset status of phases
        self.prev_phases = {}
        for intersection in self.intersections:
            current_phase = self.connection.trafficlight.getPhase(intersection)
            self.prev_phases[intersection] = current_phase
            
        self.phase0_duration = {intersection: None for intersection in self.intersections}

        return self.get_state()

    def SAC_reset(self):
        # self._cleanup_old_outputs()
        # === Reset environment state ===
        self._close_connection()
        self._open_connection()

        # === Reset EAD-related tracking ===
        self.phases_old_all = {}
        self.Qab = []
        self.last_speed = []
        self.EAD_ID_list = []
        self.ead_counts_per_step = []  # List to track EAD vehicle counts per step

        # Optionally reset vehicle stats if needed
        self.vehicle_stats = {}

        # reset status of phases
        self.prev_phases = {}
        for intersection in self.intersections:
            current_phase = self.connection.trafficlight.getPhase(intersection)
            self.prev_phases[intersection] = current_phase

        self.phase0_duration = {intersection: None for intersection in self.intersections}

        return self.get_step_state()

    def _process_action(self, a):
        action = a.copy()
        if self.single_agent and self.agent_type != 'rule':
            action = '{0:0b}'.format(a[0])
            action = action.zfill(len(self.intersections))
            action = [int(c) for c in action]
        return {intersection: action[i] for i, intersection in enumerate(self.intersections)}
    
    def _reverse_action(self, action_dict):
        actions = [action_dict[intersection] for intersection in self.intersections]
        if self.single_agent and self.agent_type != 'rule':
            # Binary to int
            bitstring = ''.join(str(bit) for bit in actions)
            return [int(bitstring, 2)]
        return actions

    def step(self, a, ep_step, get_global_reward, def_agent=False):
        action = self._process_action(a)
        
        start_gather_info_time = time.time()
        # gather vehicle level info
        if self.constants['environment']['gather_vehicle_info']:
            self._vehicle_info()
        gather_info_time = time.time() - start_gather_info_time
        self.gather_info_time += gather_info_time
        
        # control vehicle
        start_control_vehicle_time = time.time()
        if self.constants['environment']['use_vehicle_controller']:
            self._update_traffic_signal_state(ep_step)
            self._vehicle_control()
        control_vehicle_time = time.time() - start_control_vehicle_time
        self.control_vehicle_time += control_vehicle_time
        
        if not def_agent:
            real_action = self._execute_action(action) # update the real excecuted
        self.connection.simulationStep()
        s_ = self.get_state()
        r = self.get_reward(get_global_reward)
        done = False
        if self.connection.simulation.getMinExpectedNumber() <= 0 or ep_step >= self.constants['episode']['max_ep_steps']:
            if self.eval_agent:
                self._close_connection()
            else:
                print(f"[Worker {self.agent_ID}] Step: {ep_step}, Vehicles: {self.connection.simulation.getMinExpectedNumber()}")
                s_ = self.reset()
            done = True
        return s_, r, done, self._reverse_action(real_action)

    def get_state(self):
        raise NotImplementedError

    def get_step_state(self):
        raise NotImplementedError

    def get_reward(self, get_global):
        raise NotImplementedError

    def _execute_action(self, action):
        raise NotImplementedError

    def _generate_configfile(self):
        raise NotImplementedError

    def _generate_routefile(self):
        raise NotImplementedError

    def _generate_addfile(self):
        raise NotImplementedError
    
    def _update_traffic_signal_state(self, ep_step):
        raise NotImplementedError
    
    def _vehicle_control(self):
        raise NotImplementedError
    
    def _vehicle_info(self):
        raise NotImplementedError
    
    def _compute_vehicle_metrics(self):
        raise NotImplementedError