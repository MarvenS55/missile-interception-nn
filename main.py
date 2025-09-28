import numpy as np
import csv
from missile_env.environment import MissileEnv
from visualize import plot_interactive_plotly, animate_matplotlib

def adaptive_guidance(state, interceptor):
    """
    A final, adaptive guidance law that becomes more precise as it
    approaches the target to prevent overshoot and ensure an efficient intercept.
    """
    # --- Adaptive Tuning ---
    max_thrust = 750000.0  # High thrust for energy and power
    G = 9.81
    RHO = 1.225
    C_D = 0.5
    AREA = 0.0314
    target_mass = 500.0
    inter_mass = interceptor.mass  # 200
    relative_position = state[:3]
    relative_velocity = state[3:]
    
    range_to_target = np.linalg.norm(relative_position)
    if range_to_target < 1.0:
        return np.array([0.0, 0.0, 0.0])

    # --- Adaptive Chase Bbias ---
    # The strength of the "chase" command decreases as we get closer this fixes overshooting issue.
    # This transitions from a brute-force approach at long range to a
    # fine-tuned, precision approach at close range (allows for more variety when position is randomized).
    adaptive_chase_bias = max(1.0, min(40.0, range_to_target / 1000.0))

    # --- Time-to-Go Estimation ---
    closing_velocity = -np.dot(relative_velocity, relative_position / range_to_target)
    time_to_go = range_to_target / closing_velocity if closing_velocity > 0.1 else 0.1

    # Compute target current pos and vel
    target_position = interceptor.position + relative_position
    target_velocity = interceptor.velocity + relative_velocity
    # Target drag
    tgt_speed = np.linalg.norm(target_velocity)
    if tgt_speed > 0:
        tgt_drag_force = -0.5 * RHO * tgt_speed**2 * C_D * AREA * (target_velocity / tgt_speed)
    else:
        tgt_drag_force = np.array([0.0, 0.0, 0.0])
    tgt_gravity_force = np.array([0.0, 0.0, -G * target_mass])
    tgt_acc = (tgt_gravity_force + tgt_drag_force) / target_mass
    # predicted target position
    predicted_target_position = target_position + target_velocity * time_to_go + 0.5 * tgt_acc * time_to_go**2
    # required vel and acc
    required_velocity = (predicted_target_position - interceptor.position) / time_to_go
    required_acceleration = (required_velocity - interceptor.velocity) / time_to_go
    # Innnterceptor drag
    inter_speed = np.linalg.norm(interceptor.velocity)
    if inter_speed > 0:
        inter_drag_force = -0.5 * RHO * inter_speed**2 * C_D * AREA * (interceptor.velocity / inter_speed)
    else:
        inter_drag_force = np.array([0.0, 0.0, 0.0])
    inter_drag_acc = inter_drag_force / inter_mass
    # Lead command with compensations
    lead_command = required_acceleration + np.array([0.0, 0.0, G]) - inter_drag_acc

    # --- Final Hybrid Force Command ---
    chase_command = adaptive_chase_bias * (relative_position / range_to_target)
    
    commanded_acceleration = lead_command + chase_command
    force_command = commanded_acceleration * inter_mass

    # Limit the thrust (again for overshooting)
    if np.linalg.norm(force_command) > max_thrust:
        force_command = (force_command / np.linalg.norm(force_command)) * max_thrust

    return force_command

if __name__ == "__main__":
    env = MissileEnv()
    state = env.reset()
    done = False
    
    interceptor_path = [env.interceptor.position.copy()]
    target_path = [env.target.position.copy()]
    
    log_data = []
    
    step_count = 0
    max_steps = 2000
    final_status = 'In-flight'

    while not done and step_count < max_steps:
        # Using the new adaptive guidance function
        action = adaptive_guidance(state, env.interceptor)
        state, done, info = env.step(action)
        
        interceptor_path.append(env.interceptor.position.copy())
        target_path.append(env.target.position.copy())
        
        log_entry = {
            'time_step': step_count,
            'int_pos_x': env.interceptor.position[0],
            'int_pos_y': env.interceptor.position[1],
            'int_pos_z': env.interceptor.position[2],
            'tgt_pos_x': env.target.position[0],
            'tgt_pos_y': env.target.position[1],
            'tgt_pos_z': env.target.position[2],
            'applied_force_x': action[0],
            'applied_force_y': action[1],
            'applied_force_z': action[2]
        }
        log_data.append(log_entry)
        
        step_count += 1
        final_status = info['status']

    if not done:
        final_status = "FAILED: TIMEOUT"

    print(f"Simulation finished after {step_count} steps. Result: {final_status}")
    
    log_file_path = 'data/simulation_log.csv'
    with open(log_file_path, 'w', newline='') as csvfile:
        if log_data:
            fieldnames = log_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_data)
            print(f"Simulation log saved to: {log_file_path}")
        else:
            print("No data logged.")
    
    print("Displaying interactive Plotly graph...")
    plot_interactive_plotly(interceptor_path, target_path, final_status)
    
    print("Displaying Matplotlib animation...")
    animate_matplotlib(interceptor_path, target_path, final_status)
