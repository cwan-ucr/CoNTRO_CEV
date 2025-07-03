from controller.EAD_regressor import predict_velocity

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

def EAD_acceleration(veh_id, speed, last_speed, vehQs, phases_old_all, traci):
    ffv = 15
    clg = 7.5
    laneid = traci.vehicle.getLaneID(veh_id)
    tlsid = traci.vehicle.getNextTLS(veh_id)[0][0]
    dist = traci.vehicle.getNextTLS(veh_id)[0][2]

    if 20 <= dist <= 750:
        Tstart, Tend, cycle, TGy = calculate_pass_time(veh_id, tlsid, phases_old_all, traci)
        min_time = Tstart
        max_time = Tend
        for q in vehQs:
            if q[0] == tlsid and q[1] == laneid:
                min_time += (q[2] * clg) / (ffv / 2)
        round_vel = round(speed * 2) / 2
        if max_time > 0:
            if dist / max_time <= ffv:
                pred_vel = predict_velocity(min_time, max_time, round_vel, dist) - round_vel + speed
            else:
                pred_vel = predict_velocity(max_time + (cycle - TGy), max_time + cycle, round_vel, dist) - round_vel + speed
        else:
            pred_vel = speed
        traci.vehicle.setSpeed(veh_id, pred_vel)
    else:
        traci.vehicle.setSpeedMode(veh_id, 29)
        traci.vehicle.setSpeed(veh_id, -1)

    new_speed = traci.vehicle.getSpeed(veh_id)
    acc = new_speed - last_speed
    return acc, new_speed
