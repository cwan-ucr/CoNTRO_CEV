from numpy.random import choice
import random
import sys
import socket
import subprocess
import traci
import time
import os
import copy
from controller.EAD_controller_vector import EAD_acceleration_batch

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    raise EnvironmentError("Please set the SUMO_HOME environment variable to your SUMO installation path.")

from sumolib import checkBinary
import traci
import numpy as np
from environments.environment import Environment
from collections import OrderedDict
from utils.env_phases import get_phases, get_current_phase_probs
from utils.net_scrape import *

'''
Assumptions:
1. Single lane roads
2. Grid like layout, where gen or rem nodes only have one possible edge in and out
'''
# Simulation step
STEP = 1.0

# Action Bounds
CYCLE_BOUND = 0.7
GREEN_RATIO_BOUND = 0.3

# per agent
BASE_CYCLE = 40
BASE_GREEN_RATIO = 0.5

PER_AGENT_STATE_SIZE = 6
GLOBAL_STATE_SIZE = 1

SAC_AGENT_STATE_SIZE = 36
GLOBAL_SAC_STATE_SIZE = 1
# per agent
ACTION_SIZE = 2


VEH_LENGTH = 5
VEH_MIN_GAP = 2.5
DET_LENGTH_IN_CARS = 20


def get_rel_net_path(phase_id):
    return phase_id.replace('_rush_hour', '') + '.net.xml'

def get_env_name(constants):
    shape = constants['environment']['shape']
    return '{}_{}_intersections'.format(shape[0], shape[1])

class IntersectionsEnv(Environment):
    def __init__(self, constants, device, agent_ID, eval_agent, net_path, vis=False):
        super(IntersectionsEnv, self).__init__(constants, device, agent_ID, eval_agent, net_path, vis)
        # For file names
        self.env_name = get_env_name(constants)
        self.phases = get_phases(constants['environment'], net_path)
        self.node_edge_dic = get_node_edge_dict(self.net_path)
        self._generate_addfile()
        # Calc intersection distances for reward calc
        self.distances = get_cartesian_intersection_distances(net_path)
        # For vehicle info gathering and control
        self.vehicle_stats = {}  # {type: {'distance': ..., 'time': ..., 'energy': ..., 'count': ...}}
        # === EAD-related tracking
        self.phases_old_all = {}
        self.Qab = []
        self.last_speed = []  # One entry per EAD-controlled vehicle
        self.EAD_ID_list = []  # To be filled with EAD-controlled vehicle IDs during sim
        self.ead_counts_per_step = []  # List to track EAD vehicle counts per step

        # intersection cycle state
        self.prev_cycle_state = {intersection: None for intersection in self.intersections}
        self.cycle_state = {intersection: None for intersection in self.intersections}
        self.prev_cycle_action = {intersection: None for intersection in self.intersections}
        self.cycle_action = {intersection: None for intersection in self.intersections}

        self.cycle_state_origin = {intersection: [] for intersection in self.intersections} # state without encoder
        self.cycle_queue = {intersection: 0 for intersection in self.intersections}

        # intersection phases
        self.prev_phases = {}
        self.phase0_duration = {intersection: None for intersection in self.intersections}
        self.queue_last = {intersection: 0 for intersection in self.intersections}

    def _open_connection(self):
            self._generate_routefile()
            self._generate_configfile()

            sumo_binary = checkBinary('sumo-gui' if self.vis else 'sumo')
            sumo_cfg_path = f"data/{self.env_name}_{self.agent_ID}.sumocfg"
            # log_path = f"sumo_log_{self.agent_ID}.txt"

            base_port = 8813
            port = base_port + hash(self.agent_ID) % 1000
            self.port = port

            sumo_cmd = [sumo_binary, "-c", sumo_cfg_path, "--remote-port", str(port)]

            print(f"[Worker {self.agent_ID}] Launching SUMO on port {self.port}")
            print(f"[Worker {self.agent_ID}] CMD: {' '.join(sumo_cmd)}")
            self.sumo_proc = subprocess.Popen(sumo_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[Worker {self.agent_ID}] SUMO process started, waiting for connection...")

            max_retries = 10
            for i in range(max_retries):
                print(f"[Worker {self.agent_ID}] Retry {i+1} connecting to TraCI on port {self.port}")
                try:
                    self.connection = traci.connect(port=port)
                    print(f"[Worker {self.agent_ID}] Connected to TraCI!")
                    return
                except traci.exceptions.FatalTraCIError:
                    time.sleep(1)

            raise RuntimeError(f"[Worker {self.agent_ID}] Failed to connect to SUMO on port {port} after {max_retries} retries.")

    def _get_sim_step(self, normalize):
        sim_step = self.connection.simulation.getTime()
        if normalize: sim_step /= (self.constants['episode']['max_ep_steps'] / 10.)  # Normalized between 0 and 10
        return sim_step

    def initialize_signal(self, state):
        for i, intersection in enumerate(self.intersections):
            action = np.array([0.0, 0.0])
            self.cycle_state[intersection] = state[i].copy()
            self._excute_cycle_action(action, intersection, is_start=True)

    def vehicle_step(self, ep_step):

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

        self.connection.simulationStep()

        done = False
        if self.connection.simulation.getMinExpectedNumber() <= 0 or ep_step >= self.constants['episode'][
            'max_ep_steps']:
            if self.eval_agent:
                self._close_connection()
            else:
                print(
                    f"[Worker {self.agent_ID}] Step: {ep_step}, Vehicles: {self.connection.simulation.getMinExpectedNumber()}")
                s_ = self.SAC_reset()
                state = [self.signal_state_encoder(np.array(s_[i]), i) for i in range(len(s_))]
                self.initialize_signal(state)
            done = True
        return done

    def get_cycle_done(self):
        # get whether the current signal cycle is finished
        cycle_done = []
        for intersection in self.intersections:
            current_time = self.connection.simulation.getTime()
            current_phase = self.connection.trafficlight.getPhase(intersection)
            time_next_phase = self.connection.trafficlight.getNextSwitch(intersection)

            Phase_lasting = time_next_phase - current_time

            if current_phase == 3 and Phase_lasting < STEP: # the end of this cycle
                cycle_done.append(True)
            else:
                cycle_done.append(False)

        return cycle_done

    def signal_state_encoder(self, signal_state, index):
        """
        Signal state encoder
        :param signal_state:
        :param index:
        :return:
        """

        tw = 0.99 # time significance parameter

        len_steps = signal_state.__len__() if signal_state.shape.__len__() > 1 else 1
        if len_steps == 0:
            return None

        origin_state_list = np.array(signal_state)


        time_significance = np.logspace(start=0, stop=len_steps - 1, num=len_steps, base=tw).reshape(-1, 1)
        time_significance_reverse = np.flip(time_significance)

        normalized_ratio = (1 - tw ** len_steps) / (1 - tw)

        enconder_state_1 = np.cumsum(origin_state_list * time_significance, axis=0)[-1, :]
        enconder_state_2 = np.cumsum(origin_state_list * time_significance_reverse, axis=0)[-1, :]
        enconder_state_3 = np.mean(origin_state_list, axis=0) if len_steps > 1 else origin_state_list

        enconder_state = np.concatenate([enconder_state_1 / normalized_ratio,
                                         enconder_state_2 / normalized_ratio,
                                         enconder_state_3, np.array([index])])

        return enconder_state


    # todo: Work with normalization
    def get_state(self):
        # State is made of the jam length for each detector, current phase of each intersection, elapsed time
        # for the current phase of each intersection and the current ep step
        state = self._make_state()
        # Normalize if not rule
        normalize = True if self.agent_type != 'rule' else False
        # Get sim step
        sim_step = self._get_sim_step(normalize)
        for intersection, dets in list(self.intersection_dets.items()):
            # Jam length - WARNING: on sim step 0 this seems to have an issue so just return all zeros
            jam_length = [self.connection.lanearea.getJamLengthVehicle(det) for det in dets] if self.connection.simulation.getTime() != 0 else [0] * len(dets)
            self._add_to_state(state, jam_length, key='jam_length', intersection=intersection)
            # Current phase
            curr_phase = self.connection.trafficlight.getPhase(intersection)
            self._add_to_state(state, curr_phase, key='curr_phase', intersection=intersection)
            # Elapsed time of current phase
            elapsed_phase_time = self.connection.trafficlight.getPhaseDuration(intersection) - \
                               (self.connection.trafficlight.getNextSwitch(intersection) -
                                self.connection.simulation.getTime())
            if normalize: elapsed_phase_time /= 10.  # Slight normalization
            self._add_to_state(state, elapsed_phase_time, key='elapsed_phase_time', intersection=intersection)
            # Add global param of current sim step
            if not self.single_agent and self.agent_type != 'rule':
                self._add_to_state(state, sim_step, key='sim_step', intersection=intersection)
        # DeMorgan's law of above
        if self.single_agent or self.agent_type == 'rule':
            self._add_to_state(state, sim_step, key='sim_step', intersection=None)

        # Don't interpolate if single agent or agent type is rule or multiagent but state disc is 0
        if self.single_agent or self.agent_type == 'rule' or self.constants['multiagent']['state_interpolation'] == 0:
            return self._process_state(state)

        state_size = PER_AGENT_STATE_SIZE + GLOBAL_STATE_SIZE
        final_state = []
        for intersection in self.intersections:
            neighborhood = self.neighborhoods[intersection]
            # Add the intersection state itself
            intersection_state = state[intersection]
            final_state.append(np.zeros(shape=(state_size * self.max_num_neighbors,)))
            # Slice in this intersection's state not discounted
            final_state[-1][:state_size] = np.array(intersection_state)
            # Then its discounted neighbors
            for n, neighbor in enumerate(neighborhood):
                assert neighbor != intersection
                extension = self.constants['multiagent']['state_interpolation'] * np.array(state[neighbor])
                range_start = (n + 1) * state_size
                range_end = range_start + state_size
                final_state[-1][range_start:range_end] = extension
        state = self._process_state(final_state)
        return state

    def get_step_state(self):
        # State is made of the jam length, occupancy, and mean_speed of each detector
        # for the current phase of each intersection and the current ep step
        step_state = self._make_state()

        for intersection, dets in list(self.intersection_dets.items()):

            index = self.intersections.index(intersection)

            # Jam length - WARNING: on sim step 0 this seems to have an issue so just return all zeros
            Detector_LaneID = [self.connection.lanearea.getLaneID(det) for det in dets]
            Detector_length = [self.connection.lanearea.getLength(det) for det in dets]
            Max_Speed = [self.connection.lane.getMaxSpeed(LaneID) for LaneID in Detector_LaneID]

            if self.connection.simulation.getTime() != 0:
                jam_length = [self.connection.lanearea.getJamLengthMeters(det) for det in dets]
                Mean_speed = [self.connection.lanearea.getLastStepMeanSpeed(det) for det in dets]
                Occupancy = [self.connection.lanearea.getLastStepOccupancy(det) for det in dets]
            else:
                jam_length = [0] * len(dets)
                Mean_speed = [0] * len(dets)
                Occupancy = [0] * len(dets)

            self.cycle_queue[intersection] += (np.array(jam_length)).sum() / (np.array(Detector_length)).sum()

            for i, det in enumerate(dets):
                Occupancy[i] = Occupancy[i] / Detector_length[i]
                jam_length[i] = jam_length[i] / Detector_length[i]
                Mean_speed[i] = Mean_speed[i] / Max_Speed[i] if Occupancy[i] > 1e-4 else 1

            step_state[index] = Occupancy + jam_length + Mean_speed


        return step_state

    def get_cycle_reward(self, intersection):
        """
        Reward function for intersection at given index
        :param intersection:
        :return:
        """
        # Get the reward for the current cycle
        reward = np.array(-1.0 * self.cycle_queue[intersection] / BASE_CYCLE)
        return reward

    # Allows for interpolation between local and global reward given a reward disc. factor
    # get_global is used to signal returning the global rew as a single value for the eval runs, ow an array is returned
    def get_reward(self, get_global):
        reward_interpolation = self.constants['multiagent']['reward_interpolation']
        # Get local rewards for each intersection
        local_rewards = {}
        assert len(self.all_dets) > 0, "[Reward] No detectors found!"
        for intersection in self.intersections:
            # detector occupation
            dets = self.intersection_dets[intersection]
            if len(dets) == 0:
                print(f"[Warning] No detectors for intersection {intersection}, skipping reward calc.")
                local_rewards[intersection] = 0.0
                continue
            dets_rew = sum([self.connection.lanearea.getJamLengthVehicle(det) for det in dets])
            # queue length change,
            # penalty for longer queues, reward for shorter queues
            d_dets_rew = (dets_rew - self.queue_last[intersection])
            self.queue_last[intersection] = 1.0 * dets_rew

            # dets_rew = (len(dets) * DET_LENGTH_IN_CARS) - 2 * dets_rew
            # dets_rew = - dets_rew / (len(self.all_dets) * DET_LENGTH_IN_CARS)
            # todo: test performance of normalizing by len of dets NOT all dets (make sure to remove assertian below)
            # dets_rew /= (len(self.all_dets) * DET_LENGTH_IN_CARS) # w/ reward normalization
            # dets_rew /= (len(dets) * DET_LENGTH_IN_CARS) # w/o reward normalization
            # local_rewards[intersection] = dets_rew # uncomment if w/o phase_stability reward
            
            # phase_stability
            # Add phase stability
            phase = self.connection.trafficlight.getPhase(intersection)
            prev_phase = self.prev_phases[intersection]
            if phase == 1 or phase == 3 or prev_phase == 1 or prev_phase == 3:
                stability_reward = 0.0
            else:
                if phase == prev_phase:
                    stability_reward = +1.0/len(self.intersections)
                else:
                    stability_reward = -0.5/len(self.intersections)

            self.prev_phases[intersection] = phase  # update tracker
            
            # Combine rewards
            local_rewards[intersection] = (
                1.0 * d_dets_rew +
                0.01 * stability_reward
            )
            
        # If getting global then return the sum (singe value)
        # print('[Local_Rewards]', local_rewards)
        if get_global:
            ret = sum([local_rewards[i] for i in self.intersections])
            # assert -1.251*len(self.intersections) <= ret <= 1.501*len(self.intersections)
            return ret
        # if single intersection
        if len(self.intersections) == 1:
            ret = list(local_rewards.values())[0]
            # assert -1.251 <= ret <= 1.501
            return np.array([ret])
        # If single agent
        if self.single_agent:
            ret = sum([local_rewards[i] for i in self.intersections])
            # assert -1.251*len(self.intersections) <= ret <= 1.501*len(self.intersections)
            return np.array([ret])
        # Disc edge cases
        if reward_interpolation == 0.:  # Local
            ret = np.array([r for r in list(local_rewards.values())])
            return ret
        if reward_interpolation == 1.:  # global
            gr = sum([local_rewards[i] for i in self.intersections])
            ret = np.array([gr] * len(self.intersections))
            return ret
        # O.w. interpolation
        arr = []
        for intersection in self.intersections:
            dists = self.distances[intersection]
            max_dist = max([d for d in list(dists.values())])
            local_rew = 0.
            for inner_int in self.intersections:
                d = dists[inner_int]
                r = local_rewards[inner_int]
                local_rew += pow(reward_interpolation, (d / max_dist)) * r
            arr.append(local_rew)
        return np.array(arr)

    def _excute_cycle_action(self, action, intersection, is_start=False):
        """
        Execute action for given intersection index
        :param action: cycle_change_ratio and green_cycle_ratio
        :param intersection: intersection ID
        :return:
        """

        # change cycle length and green ratio with minimal constraint
        # t_cyc = t_cyc_b * (1 + delta), delta in [-0.7, 0.7], tcyc in [18, 120]
        # gre_ratio = gre_ratio_b * (1 + delta), delta in [-0.3, 0.3], gre_ratio in [0.2, 0.8], t_gre in [3, t_cyc - 9]
        cycle_change_ratio = action[0] * CYCLE_BOUND
        green_ratio = action[1] * GREEN_RATIO_BOUND
        cycle_length = int((BASE_CYCLE * (1 + cycle_change_ratio)).clip(18, 120))
        green_duration = int(((BASE_GREEN_RATIO + green_ratio) * cycle_length).clip(3, cycle_length - 9))
        yellow_duration = 3
        red_duration = cycle_length - green_duration - yellow_duration * 2

        phase_0 = "GGgsrrGGgsrr"
        phase_1 = "yyysrryyysrr"
        phase_2 = "srrGGgsrrGGg"
        phase_3 = "srryyysrryyy"

        current_phase = 3 if ~is_start else 0

        # Reset control logic
        new_logic = traci.trafficlight.Logic(
            programID="0",
            type=0,  # 0 = static
            currentPhaseIndex=current_phase,
            phases=[
                traci.trafficlight.Phase(green_duration, phase_0),
                traci.trafficlight.Phase(yellow_duration, phase_1),
                traci.trafficlight.Phase(red_duration, phase_2),
                traci.trafficlight.Phase(yellow_duration, phase_3)
            ],
            subParameter={}
        )

        self.connection.trafficlight.setProgramLogic(intersection, new_logic)
        self.connection.trafficlight.setPhaseDuration(intersection, green_duration) if is_start else None
        self.prev_cycle_state[intersection] = self.cycle_state[intersection].copy() if ~is_start else None
        self.cycle_state_origin[intersection] = []
        self.cycle_queue[intersection] = 0

    # Switch
    # action: {"intersectionNW": 0 or 1, .... }
    def _execute_action(self, action):
        real_action = {intersection: 0 for intersection in self.intersections}
        def _get_current_phase_elapsed_time(tls_id):
            duration = self.connection.trafficlight.getPhaseDuration(tls_id)
            time_to_switch = self.connection.trafficlight.getNextSwitch(tls_id) - self.connection.simulation.getTime()
            elapsed = duration - time_to_switch
            return max(0.0, elapsed)  # avoid negative
        # dont allow ANY switching if in yellow phase (ie in process of switching)
        # Loop through digits, one means switch, zero means stay
        
        if self.constants["agent"]["agent_type"] == 'ppo':
            for intersection in self.intersections:
                value = action[intersection]
                currPhase = self.connection.trafficlight.getPhase(intersection)
                currTime = _get_current_phase_elapsed_time(intersection)

                # if value == 0:
                #     continue
                
                if currPhase == 1 or currPhase == 3:  # Yellow, pass
                    continue

                # Switch from Phase 0
                if value == 1 and currPhase == 0 and 3 <= currTime <= 30:
                    self.phase0_duration[intersection] = currTime  # Record duration of phase 0
                    newPhase = currPhase + 1
                    self.connection.trafficlight.setPhase(intersection, newPhase)
                    real_action[intersection] = 1

                # Switch from Phase 2
                if currPhase == 2:
                    duration_phase0 = self.phase0_duration.get(intersection, None)
                    if value == 1 and duration_phase0 is not None and 3 <= currTime < (54 - duration_phase0):
                        newPhase = currPhase + 1
                        self.connection.trafficlight.setPhase(intersection, newPhase)
                        real_action[intersection] = 1
                    elif value == 0 and duration_phase0 is not None and currTime == (114 - duration_phase0):
                        newPhase = currPhase + 1
                        self.connection.trafficlight.setPhase(intersection, newPhase)
                        real_action[intersection] = 1
                        
        elif self.constants["agent"]["agent_type"] == 'rule':
            for intersection in self.intersections:
                value = action[intersection]
                currPhase = self.connection.trafficlight.getPhase(intersection)
                if currPhase == 1 or currPhase == 3:  # Yellow, pass
                    continue
                if value == 0:  # do nothing
                    continue
                else:  # switch
                    newPhase = currPhase + 1
                    self.connection.trafficlight.setPhase(intersection, newPhase)
        
        return real_action

    def _generate_configfile(self):
        config_path = f'data/{self.env_name}_{self.agent_ID}.sumocfg'
        config_content = f"""<?xml version="1.0" encoding="UTF-8"?>
    <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
        <input>
            <net-file value="{get_rel_net_path(self.env_name)}"/>
            <route-files value="{self.env_name}_{self.agent_ID}.rou.xml"/>
            <additional-files value="{self.env_name}_{self.agent_ID}.add.xml"/>
        </input>

        <time>
            <begin value="0"/>
        </time>

        <report>
            <verbose value="false"/>
            <no-step-log value="true"/>
            <no-warnings value="true"/>
        </report>
    </configuration>
    """
        try:
            with open(config_path, 'w') as config:
                config.write(config_content)
        except Exception as e:
            print(f"[Worker {self.agent_ID}] Error writing SUMO config file: {e}")


    def _add_vehicle(self, t, node_probs):
        trips = []
        rem_nodes = list(node_probs.keys())
        rem_probs = [v['rem'] for v in node_probs.values()]

        def pick_rem_edge(gen_node):
            while True:
                chosen = choice(rem_nodes, p=rem_probs)
                if chosen != gen_node:
                    return chosen

        for gen_k, dic in node_probs.items():
            if random.random() < dic['gen']:
                rem_k = pick_rem_edge(gen_k)
                route_id = f"{gen_k}___{rem_k}"
                gen_edge = self.node_edge_dic[gen_k]['gen']
                rem_edge = self.node_edge_dic[rem_k]['rem']
                trips.append(f'    <trip id="{route_id}_{t}" type="car" from="{gen_edge}" to="{rem_edge}" depart="{t}"/>')

        return trips


    def _generate_routefile(self):
        route_lines = [
            "<routes>",
            f'    <vType id="car" accel="0.8" decel="4.5" sigma="0.5" length="{VEH_LENGTH}" minGap="{VEH_MIN_GAP}" maxSpeed="15" guiShape="passenger"/>'
        ]
        
        for t in range(self.constants['episode']['generation_ep_steps']):
            trips = self._add_vehicle(t, get_current_phase_probs(t, self.phases, self.constants['episode']['generation_ep_steps']))
            route_lines.extend(trips)

        route_lines.append("</routes>")

        route_path = f"data/{self.env_name}_{self.agent_ID}.rou.xml"
        with open(route_path, "w") as f:
            f.write("\n".join(route_lines))


    def _generate_addfile(self):
        from xml.sax.saxutils import escape

        self.all_dets = []
        self.intersection_dets = OrderedDict({k: [] for k in self.intersections})

        add_lines = ['<additionals>']

        tree = ET.parse(self.net_path)
        root = tree.getroot()

        for edge in root.iter('edge'):
            edge_id = edge.attrib['id']
            if 'function' in edge.attrib or 'intersection' not in edge.attrib['to']:
                continue
            length = float(edge[0].attrib['length'])
            pos = length - (DET_LENGTH_IN_CARS * (VEH_LENGTH + VEH_MIN_GAP))
            det_id = f'DET+++{edge_id}'
            self.all_dets.append(det_id)
            self.intersection_dets[edge.attrib['to']].append(det_id)

            add_lines.append(
                f'    <e2Detector id="{escape(det_id)}" lane="{escape(edge_id)}_0" pos="{pos}" endPos="{length}" '
                f'freq="100000" friendlyPos="true" file="{escape(self.env_name)}.out"/>'
            )

        add_lines.append(f'    <edgeData id="edgeData_0" file="edgeData_{self.agent_ID}.out.xml"/>')
        add_lines.append('</additionals>')

        add_path = f"data/{self.env_name}_{self.agent_ID}.add.xml"
        with open(add_path, "w") as f:
            f.write("\n".join(add_lines))
            
    def _update_traffic_signal_state(self, ep_step):
        """
        Precompute signal phase structure and lane queues.
        Call this once per simulation step before _vehicle_control().
        """
        tls_ids = self.connection.trafficlight.getIDList()

        # === 1. Record signal phases (every 200 steps only)
        if ep_step % 200 == 0:
            self.phases_old_all = {
                tls_id: logic.phases
                for tls_id in tls_ids
                for logic in self.connection.trafficlight.getAllProgramLogics(tls_id)
                if logic.programID == self.connection.trafficlight.getProgram(tls_id)
            }

        # === 2. Update actuated signal durations
        if self.constants['environment']['actuated_tls']:
            for tls_id in tls_ids:
                if 'G' in self.connection.trafficlight.getRedYellowGreenState(tls_id):
                    if self.connection.trafficlight.getSpentDuration(tls_id) == 1:
                        current_phase = self.connection.trafficlight.getPhase(tls_id)
                        next_switch = self.connection.trafficlight.getNextSwitch(tls_id)
                        current_time = self.connection.simulation.getTime()
                        self.phases_old_all[tls_id][current_phase].duration = (
                            next_switch - current_time + 1
                        )

        # === 3. Update queue lengths and EAD vehicle tracking
        self.Qab = []
        ead_set = set(self.EAD_ID_list)  # Use set for faster lookup

        for tls_id in tls_ids:
            controlled_lanes = self.connection.trafficlight.getControlledLanes(tls_id)
            for lane in controlled_lanes:
                veh_ids = self.connection.lane.getLastStepVehicleIDs(lane)
                queue_len = 0

                for veh_id in veh_ids:
                    # EAD vehicle registration
                    if self.connection.vehicle.getTypeID(veh_id) == "car" and veh_id not in ead_set:
                        self.EAD_ID_list.append(veh_id)
                        self.last_speed.append(13.0)
                        ead_set.add(veh_id)

                    # Count queued vehicles
                    if self.connection.vehicle.getSpeed(veh_id) < 0.1:
                        tls_info = self.connection.vehicle.getNextTLS(veh_id)
                        if tls_info and tls_info[0][2] < 150:
                            queue_len += 1

                self.Qab.append([tls_id, lane, queue_len])

         
    def _vehicle_control(self):
        """
        Vector-based vehicle control for EAD vehicles.
        """
        ead_ids = []
        speeds = []
        last_speeds = []

        ead_index_map = {veh_id: idx for idx, veh_id in enumerate(self.EAD_ID_list)}

        for tls_id in self.connection.trafficlight.getIDList():
            for lane in self.connection.trafficlight.getControlledLanes(tls_id):
                for veh_id in self.connection.lane.getLastStepVehicleIDs(lane):
                    idx = ead_index_map.get(veh_id, None)
                    if idx is not None and self.connection.vehicle.getNextTLS(veh_id):
                        ead_ids.append(veh_id)
                        speeds.append(self.connection.vehicle.getSpeed(veh_id))
                        last_speeds.append(self.last_speed[idx])

        # calculate speed and acc in batch
        if ead_ids:
            accs, new_speeds = EAD_acceleration_batch(
                ead_ids, speeds, last_speeds,
                self.Qab, self.phases_old_all,
                self.connection
            )

            # update state and process car-following logic
            for i, veh_id in enumerate(ead_ids):
                idx = ead_index_map[veh_id]
                self.last_speed[idx] = new_speeds[i]

                leader = self.connection.vehicle.getLeader(veh_id)
                if leader and leader[1] < 20:
                    self.connection.vehicle.setSpeed(veh_id, -1)
                    
        # Track the number of EAD vehicles this step
        # self.ead_counts_per_step.append(len(ead_ids))
        # print(self.ead_counts_per_step)


    
    def _vehicle_info(self):
        """
        Lower-level vehicle info ('distance', 'time', 'energy','veh_num') gathering
        """
        
        vehicle_ids = self.connection.vehicle.getIDList()
        for vid in vehicle_ids:
            # vehicle info gathering
            vtype = self.connection.vehicle.getTypeID(vid)
            speed = self.connection.vehicle.getSpeed(vid)
            energy = self.connection.vehicle.getFuelConsumption(vid)  # or getElectricConsumption() for EVs

            if vtype not in self.vehicle_stats:
                self.vehicle_stats[vtype] = {'distance': 0.0, 'time': 0.0, 'energy': 0.0, 'veh_num': 0.0}

            self.vehicle_stats[vtype]['distance'] += speed * self.constants['environment']['step_length']
            self.vehicle_stats[vtype]['time'] += self.constants['environment']['step_length']
            self.vehicle_stats[vtype]['energy'] += energy * self.constants['environment']['step_length']
            self.vehicle_stats[vtype]['veh_num'] += 1
         
            
    def _compute_vehicle_metrics(self):
        result = {}
        for vtype, stats in self.vehicle_stats.items():
            distance = stats['distance']
            time = stats['time']
            energy = stats['energy']
            veh_num = stats['veh_num']
            steps_num = self.constants['episode']['max_ep_steps']
            result[f'avg_speed_{vtype.lower()}'] = (distance / time) if time > 0 else 0
            result[f'energy_per_mile_{vtype.lower()}'] = (energy / distance) if distance > 0 else 0
            result[f'avg_veh_num_{vtype.lower()}'] = veh_num/steps_num if steps_num > 0 else 0
        return result                