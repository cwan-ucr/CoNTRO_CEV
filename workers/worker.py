import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
from copy import deepcopy
import os
import time


# Parent - abstract
class Worker:
    def __init__(self, constants, device, env, id, data_collector):
        self.constants = constants
        self.device = device
        self.env = env
        self.id = id
        self.data_collector = data_collector

    def _reset(self):
        raise NotImplementedError

    def _get_prediction(self, states, actions=None, ep_step=None):
        raise NotImplementedError

    def _get_action(self, prediction):
        raise NotImplementedError

    def _copy_shared_model_to_local(self):
        raise NotImplementedError

    # This method gets the ep results i really care about (gets averages)
    def _read_edge_results(self, file, results):
        tree = ET.parse(file)
        root = tree.getroot()
        for c in root.iter('edge'):
            for k, v in list(c.attrib.items()):
                if k == 'id': continue
                results[k].append(float(v))
        return results

    # ep_count only for test
    def eval_episode(self, results):
        ep_rew = 0
        step = 0
        state = self.env.reset()
        self._reset()

        while True:
            prediction = self._get_prediction(state, ep_step=step)
            action = self._get_action(prediction)
            next_state, reward, done, _ = self.env.step(action, step, get_global_reward=True)

            if isinstance(reward, (np.ndarray, list)):
                if np.any(np.isnan(reward)) or np.any(np.isinf(reward)):
                    print(f"[Worker {self.env.agent_ID}] Invalid reward array in eval: {reward}")
                    reward = 0
            elif np.isnan(reward) or np.isinf(reward):
                print(f"[Worker {self.env.agent_ID}] Invalid scalar reward in eval: {reward}")
                reward = 0

            if not (np.isnan(ep_rew) or np.isinf(ep_rew)):
                ep_rew += reward
            else:
                print(f"[Worker {self.env.agent_ID}] Skipping reward accumulation due to inf/nan in ep_rew")

            if done:
                break

            state = np.copy(next_state) if not isinstance(state, dict) else deepcopy(next_state)
            step += 1

        # === XML parsing safety ===
        xml_path = f'data/edgeData_{self.id}.out.xml'
        for _ in range(10):
            if os.path.exists(xml_path) and os.path.getsize(xml_path) > 100:
                try:
                    ET.parse(xml_path)
                    results = self._read_edge_results(xml_path, results)
                    break
                except ET.ParseError as e:
                    print(f"[Worker {self.env.agent_ID}] XML parse error: {e}")
            time.sleep(0.2)
        else:
            print(f"[Worker {self.env.agent_ID}] Warning: Skipping XML parsing due to invalid or missing file.")

        if np.isnan(ep_rew) or np.isinf(ep_rew):
            print(f"[Worker {self.env.agent_ID}] Eval episode reward is invalid: {ep_rew}")
            ep_rew = 0
            
        # Collect additional vehicle-level metrics here
        if self.constants['environment']['gather_vehicle_info']:
            veh_metrics = self.env._compute_vehicle_metrics()  # You define this in env
            for k, v in veh_metrics.items():
                results[k].append(v)
                
        if not np.isnan(ep_rew) and not np.isinf(ep_rew):
            results['rew'].append(ep_rew)
        else:
            print(f"[Worker {self.env.agent_ID}] Skipping invalid ep_rew in results.")

        return results


    def eval_episodes(self, current_rollout, model_state=None, ep_count=None):
        print(f"[Worker {self.env.agent_ID}] >>> ENTERED eval_episodes <<< current rollout: {current_rollout}")
        self._copy_shared_model_to_local()
        results = defaultdict(list)
        
        for ep in range(self.constants['episode']['eval_num_eps']):
            results = self.eval_episode(results)

        # Optional debug print of raw results
        # for k, v_list in results.items():
        #     print(f"[Worker {self.env.agent_ID}] Raw {k} values: {v_list}")

        # Optional: mark the rollout number in results
        if current_rollout:
            results['rollout'] = [current_rollout]

        # Safely average valid values only
        filtered_results = {}
        for k, v_list in results.items():
            valid = [x for x in v_list if not np.isnan(x) and not np.isinf(x)]
            if not valid:
                print(f"[Worker {self.env.agent_ID}] Warning: All values in {k} are invalid (NaN/Inf).")
                filtered_results[k] = float('nan')  # or use 0.0, or skip storing
            else:
                filtered_results[k] = sum(valid) / len(valid)
                

        self.data_collector.collect_ep(
            filtered_results,
            model_state,
            ep_count + self.constants['episode']['eval_num_eps'] if ep_count is not None else None
        )
        
        print(f"[Worker {self.env.agent_ID}] Finished eval_episodes!", flush=True)


