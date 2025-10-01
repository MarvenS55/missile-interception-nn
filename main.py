import numpy as np
import h5py
from missile_env.environment import MissileEnv

def adaptive_guidance(state, interceptor):
    """Conservative APN guidance law used to generate expert data."""
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
        return np.zeros(3)
    
    LOS_unit = relative_position / r
    V_closing = -np.dot(relative_velocity, LOS_unit)
    if V_closing <= 0.1:
        V_closing = max(0.1, r / 300.0)
    
    omega = np.cross(LOS_unit, relative_velocity) / r
    pn_command = N * V_closing * np.cross(omega, LOS_unit)
    
    target_velocity = interceptor.velocity + relative_velocity
    tgt_speed = np.linalg.norm(target_velocity)
    tgt_drag_force = -0.5 * RHO * tgt_speed**2 * C_D * AREA * (target_velocity / tgt_speed) if tgt_speed > 0 else np.zeros(3)
    tgt_gravity_force = np.array([0.0, 0.0, -G * target_mass])
    a_target = (tgt_gravity_force + tgt_drag_force) / target_mass
    
    a_t_proj = np.dot(a_target, LOS_unit)
    a_t_perp = a_target - a_t_proj * LOS_unit
    aug_command = (N / 2.0) * a_t_perp
    
    commanded_acceleration = pn_command + aug_command
    
    inter_speed = np.linalg.norm(interceptor.velocity)
    inter_drag_force = -0.5 * RHO * inter_speed**2 * C_D * AREA * (interceptor.velocity / inter_speed) if inter_speed > 0 else np.zeros(3)
    inter_drag_acc = inter_drag_force / inter_mass
    commanded_acceleration += np.array([0.0, 0.0, G]) - inter_drag_acc
    
    force_command = commanded_acceleration * inter_mass
    
    thrust_norm = np.linalg.norm(force_command)
    if thrust_norm > max_thrust:
        force_command = (force_command / thrust_norm) * max_thrust
        
    return force_command

if __name__ == "__main__":
    num_simulations = 5000
    max_steps = 2000
    
    results = []
    
    h5_file = 'data/dataset.h5'
    with h5py.File(h5_file, 'w') as hf:
        # Pre create datasets for appending
        states_ds = hf.create_dataset('states', shape=(0, 6), maxshape=(None, 6), dtype='float32', compression='gzip')
        actions_ds = hf.create_dataset('actions', shape=(0, 3), maxshape=(None, 3), dtype='float32', compression='gzip')
        next_states_ds = hf.create_dataset('next_states', shape=(0, 6), maxshape=(None, 6), dtype='float32', compression='gzip')
        rewards_ds = hf.create_dataset('rewards', shape=(0,), maxshape=(None,), dtype='float32', compression='gzip')
        dones_ds = hf.create_dataset('dones', shape=(0,), maxshape=(None,), dtype='bool', compression='gzip')
        
        total_samples = 0
        for run_id in range(num_simulations):
            env = MissileEnv()
            state = env.reset()
            done = False
            
            episode_states, episode_actions, episode_next_states, episode_rewards, episode_dones = [], [], [], [], []
            
            step_count = 0
            final_status = 'In-flight'
            while not done and step_count < max_steps:
                action = adaptive_guidance(state, env.interceptor)
                prev_state = state.copy()
                state, done, info = env.step(action)
                
                distance = np.linalg.norm(state[:3])
                reward = -distance / 1000.0
                if info['status'] == 'Intercept':
                    reward += 100
                elif 'FAILED' in info['status']:
                    reward -= 100
                
                episode_states.append(prev_state)
                episode_actions.append(action)
                episode_next_states.append(state)
                episode_rewards.append(reward)
                episode_dones.append(done)
                
                step_count += 1
                final_status = info['status']
            
            if not done:
                final_status = "FAILED: TIMEOUT"

            results.append({'status': final_status})

            # append episode data to HDF5
            ep_len = len(episode_states)
            states_ds.resize((total_samples + ep_len, 6))
            states_ds[total_samples:] = np.array(episode_states, dtype='float32')
            actions_ds.resize((total_samples + ep_len, 3))
            actions_ds[total_samples:] = np.array(episode_actions, dtype='float32')
            next_states_ds.resize((total_samples + ep_len, 6))
            next_states_ds[total_samples:] = np.array(episode_next_states, dtype='float32')
            rewards_ds.resize((total_samples + ep_len,))
            rewards_ds[total_samples:] = np.array(episode_rewards, dtype='float32')
            dones_ds.resize((total_samples + ep_len,))
            dones_ds[total_samples:] = np.array(episode_dones, dtype='bool')
            
            total_samples += ep_len
            print(f"Run {run_id} completed: {final_status}, total samples: {total_samples}")

    # final stats for the classical guidance
    num_success = sum(1 for res in results if res['status'] == 'Intercept')
    success_rate = num_success / num_simulations * 100
    print(f"\n--- Classical Guidance Stats ---")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Dataset saved to {h5_file}")