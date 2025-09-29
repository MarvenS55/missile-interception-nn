import numpy as np
import csv
from missile_env.environment import MissileEnv
from visualize import plot_interactive_plotly

def adaptive_guidance(state, interceptor):
    """
    Conservative APN guidance that maintains the working logic
    """
    N = 4.0
    max_thrust = 3000000.0
    G = 9.81
    RHO = 1.225
    C_D = 0.5
    AREA = 0.0314
    target_mass = 500.0
    inter_mass = interceptor.mass
    
    relative_position = state[:3]
    relative_velocity = state[3:]
    
    r = np.linalg.norm(relative_position)
    if r < 1.0:
        return np.array([0.0, 0.0, 0.0]), 0.0
    
    LOS_unit = relative_position / r
    
    # Closing velocity
    V_closing = -np.dot(relative_velocity, LOS_unit)
    if V_closing <= 0.1:
        V_closing = max(0.1, r / 300.0)
    
    # LOS rate
    omega = np.cross(LOS_unit, relative_velocity) / r
    
    # Base PN command
    pn_command = N * V_closing * np.cross(omega, LOS_unit)
    
    # Target acceleration estimation (needs tweaking)
    target_velocity = interceptor.velocity + relative_velocity
    tgt_speed = np.linalg.norm(target_velocity)
    if tgt_speed > 0:
        tgt_drag_force = -0.5 * RHO * tgt_speed**2 * C_D * AREA * (target_velocity / tgt_speed)
    else:
        tgt_drag_force = np.array([0.0, 0.0, 0.0])
    tgt_gravity_force = np.array([0.0, 0.0, -G * target_mass])
    a_target = (tgt_gravity_force + tgt_drag_force) / target_mass
    
    # APN augmentation
    a_t_proj = np.dot(a_target, LOS_unit)
    a_t_perp = a_target - a_t_proj * LOS_unit
    aug_command = (N / 2.0) * a_t_perp
    
    # Total commanded acceleration
    commanded_acceleration = pn_command + aug_command
    
    # Simple energy management - only minimal upward bias when really needed
    altitude_difference = relative_position[2]
    if altitude_difference > 5000 and interceptor.position[2] < 10000:
        # Mild upward bias for high targets when we're low
        commanded_acceleration[2] += 0.5 * G
    
    # Compensate for interceptor dynamics
    inter_speed = np.linalg.norm(interceptor.velocity)
    if inter_speed > 0:
        inter_drag_force = -0.5 * RHO * inter_speed**2 * C_D * AREA * (interceptor.velocity / inter_speed)
    else:
        inter_drag_force = np.array([0.0, 0.0, 0.0])
    inter_drag_acc = inter_drag_force / inter_mass
    commanded_acceleration += np.array([0.0, 0.0, G]) - inter_drag_acc
    
    # Force command
    force_command = commanded_acceleration * inter_mass
    
    # Limit thrust
    thrust_norm = np.linalg.norm(force_command)
    if thrust_norm > max_thrust:
        force_command = (force_command / thrust_norm) * max_thrust
    
    # Simple timetoo-go
    time_to_go = r / max(V_closing, 0.1)
    
    return force_command, time_to_go

if __name__ == "__main__":
    num_simulations = 1000
    visualize = False
    max_steps = 2000  
    
    results = []
    all_log_data = []
    
    for run_id in range(num_simulations):
        env = MissileEnv()
        state = env.reset()
        print(f"Run {run_id}: Target at {env.target.position}, velocity: {env.target.velocity}")
        done = False
        
        interceptor_path = [env.interceptor.position.copy()]
        target_path = [env.target.position.copy()]
        
        log_data = []
        step_count = 0
        final_status = 'In-flight'

        while not done and step_count < max_steps:
            action, time_to_go = adaptive_guidance(state, env.interceptor)
            state, done, info = env.step(action)
            
            interceptor_path.append(env.interceptor.position.copy())
            target_path.append(env.target.position.copy())
            
            relative_position = state[:3]
            relative_velocity = state[3:]
            rel_dist = np.linalg.norm(relative_position)
            closing_vel = -np.dot(relative_velocity, relative_position / rel_dist) if rel_dist > 0 else 0
            
            log_entry = {
                'run_id': run_id,
                'time_step': step_count,
                'int_pos_x': env.interceptor.position[0],
                'int_pos_y': env.interceptor.position[1],
                'int_pos_z': env.interceptor.position[2],
                'int_vel_x': env.interceptor.velocity[0],
                'int_vel_y': env.interceptor.velocity[1],
                'int_vel_z': env.interceptor.velocity[2],
                'tgt_pos_x': env.target.position[0],
                'tgt_pos_y': env.target.position[1],
                'tgt_pos_z': env.target.position[2],
                'tgt_vel_x': env.target.velocity[0],
                'tgt_vel_y': env.target.velocity[1],
                'tgt_vel_z': env.target.velocity[2],
                'applied_force_x': action[0],
                'applied_force_y': action[1],
                'applied_force_z': action[2],
                'relative_distance': rel_dist,
                'closing_velocity': closing_vel,
                'estimated_time_to_go': time_to_go
            }
            log_data.append(log_entry)
            all_log_data.append(log_entry)
            
            step_count += 1
            final_status = info['status']

        if not done:
            final_status = "FAILED: TIMEOUT"
        
        final_distance = np.linalg.norm(env.target.position - env.interceptor.position)
        print(f"Run {run_id}: {step_count} steps. Result: {final_status}. Final distance: {final_distance:.2f} m")
        
        results.append({
            'run_id': run_id,
            'status': final_status,
            'steps': step_count,
            'final_distance': final_distance
        })
    
    # Save logs
    log_file_path = 'data/simulation_log.csv'
    with open(log_file_path, 'w', newline='') as csvfile:
        if all_log_data:
            fieldnames = all_log_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_log_data)
    
    # Statistics
    num_success = sum(1 for res in results if res['status'] == 'Intercept')
    success_rate = num_success / num_simulations * 100
    
    success_steps = [res['steps'] for res in results if res['status'] == 'Intercept']
    avg_steps_success = np.mean(success_steps) if success_steps else 0
    
    fail_distances = [res['final_distance'] for res in results if res['status'] != 'Intercept']
    avg_miss_dist = np.mean(fail_distances) if fail_distances else 0
    
    failure_types = {}
    for res in results:
        if res['status'] != 'Intercept':
            failure_types[res['status']] = failure_types.get(res['status'], 0) + 1
    
    print("\n--- Simulation Statistics ---")
    print(f"Total runs: {num_simulations}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Average steps for successful intercepts: {avg_steps_success:.2f}")
    print(f"Average miss distance for failures: {avg_miss_dist:.2f} m")
    
    if failure_types:
        print("\n--- Failure Analysis ---")
        for failure_type, count in failure_types.items():
            print(f"{failure_type}: {count} occurrences")
