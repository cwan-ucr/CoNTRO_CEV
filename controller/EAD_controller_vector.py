from controller.EAD_regressor_vector import predict_velocity_vectorized
import numpy as np

def calculate_pass_time(veh_id, tlsid, phases_old_all, traci):
    tls_index = traci.vehicle.getNextTLS(veh_id)[0][1]
    vehicle_state = traci.vehicle.getNextTLS(veh_id)[0][3]
    phases = phases_old_all[tlsid]
    current_phase = traci.trafficlight.getPhase(tlsid)
    next_switch = traci.trafficlight.getNextSwitch(tlsid)
    current_time = traci.simulation.getTime()
    cycle = sum(phase.duration for phase in phases)

    Tstart = Tend = TGy = None
    if vehicle_state in 'GgYy':
        Tstart = 0
        remaining = next_switch - current_time
        time_to_red = sum(
            phases[i % len(phases)].duration for i in range(current_phase + 1, current_phase + len(phases))
            if 'r' not in phases[i % len(phases)].state[tls_index]
        )
        Tend = time_to_red + remaining
        time_from_red = sum(
            phases[i % len(phases)].duration for i in range(current_phase - 1, current_phase - len(phases) - 1, -1)
            if 'r' not in phases[i % len(phases)].state[tls_index]
        )
        TGy = time_from_red + phases[current_phase].duration + time_to_red
    elif vehicle_state in 'rs':
        remaining = next_switch - current_time
        time_to_green = 0
        for i in range(current_phase + 1, current_phase + len(phases) + 1):
            idx = i % len(phases)
            state = phases[idx].state
            if 'G' in state[tls_index] or 'g' in state[tls_index]:
                Tstart = time_to_green + remaining
                duration_green = phases[idx].duration
                Tend = Tstart + sum(
                    phases[j % len(phases)].duration for j in range(i + 1, current_phase + len(phases) + 1)
                    if 'r' in phases[j % len(phases)].state[tls_index]
                )
                break
            time_to_green += phases[idx].duration
        TGy = Tend - Tstart
    else:
        # Unknown state
        print(f"[Warning] Vehicle {veh_id} at tls {tlsid} has unknown state '{vehicle_state}'")
        return None, None, cycle, None
    return Tstart, Tend, cycle, TGy

def EAD_acceleration_batch(veh_ids, speeds, last_speeds, vehQs, phases_old_all, traci):
    ffv = 15
    clg = 7.5
    
    min_time_arr = []
    max_time_arr = []
    init_vel_arr = []
    dist_arr = []
    veh_id_valid = []
    veh_idx_map = {}

    for idx, veh_id in enumerate(veh_ids):
        
        if not traci.vehicle.getNextTLS(veh_id):
            print(f"[Batch] {veh_id} skipped: no TLS ahead")
            continue 
        laneid = traci.vehicle.getLaneID(veh_id)
        tlsid = traci.vehicle.getNextTLS(veh_id)[0][0]
        dist = traci.vehicle.getNextTLS(veh_id)[0][2]

        if 20 <= dist <= 750:
            Tstart, Tend, cycle, TGy = calculate_pass_time(veh_id, tlsid, phases_old_all, traci)
            if Tstart is None or Tend is None:
                continue

            min_time = Tstart
            for q in vehQs:
                if q[0] == tlsid and q[1] == laneid:
                    min_time += (q[2] * clg) / (ffv / 2)

            round_vel = round(speeds[idx] * 2) / 2
            if Tend > 0:
                if dist / Tend <= ffv:
                    min_time_arr.append(min_time)
                    max_time_arr.append(Tend)
                else:
                    min_time_arr.append(Tend + (cycle - TGy))
                    max_time_arr.append(Tend + cycle)
            else:
                continue

            dist_arr.append(dist)
            init_vel_arr.append(round_vel)
            veh_id_valid.append(veh_id)
            veh_idx_map[veh_id] = idx
            
            # print(f"[Debug] veh {veh_id} dist={dist}, min={min_time}, max={Tend}")


    # Predict velocities
    if len(min_time_arr) > 0:
        pred_vels = predict_velocity_vectorized(
            np.array(min_time_arr),
            np.array(max_time_arr),
            np.array(init_vel_arr),
            np.array(dist_arr)
        )
        

        for i, veh_id in enumerate(veh_id_valid):
            idx = veh_idx_map[veh_id]
            pred_speed = pred_vels[i] - init_vel_arr[i] + speeds[idx]
            traci.vehicle.setSpeed(veh_id, pred_speed)

    # Vehicles outside EAD range: reset control
    for idx, veh_id in enumerate(veh_ids):
        if not traci.vehicle.getNextTLS(veh_id):
            continue
        dist = traci.vehicle.getNextTLS(veh_id)[0][2]
        if dist < 20 or dist > 750:
            traci.vehicle.setSpeedMode(veh_id, 29)
            traci.vehicle.setSpeed(veh_id, -1)

    # Return acceleration and new speeds
    accs = []
    new_speeds = []
    for idx, veh_id in enumerate(veh_ids):
        speed_now = traci.vehicle.getSpeed(veh_id)
        accs.append(speed_now - last_speeds[idx])
        new_speeds.append(speed_now)
    return accs, new_speeds