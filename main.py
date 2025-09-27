import numpy as np
from missile_env.environment import MissileEnv
# FIX: Imported both visualization functions
from visualize import plot_interactive_plotly, animate_matplotlib

def augmented_proportional_navigation(state, interceptor_velocity):
    """
    A more advanced augmented prop Navigation.
    This combines a chase component with a lead compoonent so it doesnt just mirror thee object.
    """
    N = 4.5
    max_thrust = 50000.0
    chase_bias = 40.0

    relative_position = state[:3]
    relative_velocity = state[3:]
    
    range_to_target = np.linalg.norm(relative_position)
    if range_to_target < 1.0: return np.array([0.0, 0.0, 0.0])

    los_unit_vec = relative_position / range_to_target
    los_rotation_rate_vec = np.cross(relative_position, relative_velocity) / (range_to_target**2)
    closing_velocity = -np.dot(relative_velocity, los_unit_vec)
    lead_acceleration = N * closing_velocity * np.cross(los_rotation_rate_vec, los_unit_vec)
    chase_acceleration = chase_bias * los_unit_vec
    commanded_acceleration = lead_acceleration + chase_acceleration
    force_command = commanded_acceleration

    if np.linalg.norm(force_command) > max_thrust:
        force_command = (force_command / np.linalg.norm(force_command)) * max_thrust

    return force_command

if __name__ == "__main__":
    env = MissileEnv()
    state = env.reset()
    done = False
    
    interceptor_path = [env.interceptor.position.copy()]
    target_path = [env.target.position.copy()]
    
    step_count = 0
    max_steps = 2000
    final_status = 'In-flight'

    while not done and step_count < max_steps:
        action = augmented_proportional_navigation(state, env.interceptor.velocity)
        state, done, info = env.step(action)
        
        interceptor_path.append(env.interceptor.position.copy())
    
        target_path.append(env.target.position.copy())
        
        step_count += 1
        final_status = info['status']

    if not done:
        final_status = "FAILED: TIMEOUT"

    print(f"Simulation finished after {step_count} steps. Result: {final_status}")
    
    # --- Calling both visualizations plotyly better but matplot has animation ---
    print("Displaying interactive Plotly graph...")
    plot_interactive_plotly(interceptor_path, target_path, final_status)
    
    print("Displaying Matplotlib animation...")
    animate_matplotlib(interceptor_path, target_path, final_status)

