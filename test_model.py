import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from missile_env.environment import MissileEnv

# Load the V2 model and stats
model_path = os.path.join("models", "ppo_missile_model_v2")
stats_path = os.path.join("models", "vec_normalize_v2.pkl")

# Sset up the evaluation environment
eval_env = DummyVecEnv([lambda: MissileEnv()])
eval_env = VecNormalize.load(stats_path, eval_env)
eval_env.training = False
eval_env.norm_reward = False
model = PPO.load(model_path, env=eval_env)

if __name__ == "__main__":
    num_simulations = 1000 # % success stays consisten from 100 so no need for this much runs but its ok
    
    results = []
    
    for run_id in range(num_simulations):
        # Set a different seed for each run so you arent having common scenarious
        eval_env.seed(42 + run_id)
        
        obs = eval_env.reset()
        done = [False]  # iinitialize as list to match vectorized env format
        
        step_count = 0
        final_status = 'In-flight'
        
        while not done[0]:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            
            # Only collect data if episode is not done
            if not done[0]:
                # not storing paths, just counting steps
                pass
            else:
                # episode is done , record final status and break immediately
                final_status = info[0]['status']
                break
            
            step_count += 1
            
            # Safety check to prevent infinite loops, usually means high altitude missile + long chase
            if step_count > 10000:
                final_status = 'FAILED: TIMEOUT'
                break
        
        results.append({'status': final_status, 'steps': step_count})
        
        # Progress indicator for long runs so i know it didnt time out
        if (run_id + 1) % 100 == 0:
            num_success_so_far = sum(1 for res in results if res['status'] == 'Intercept')
            print(f"Completed {run_id + 1}/{num_simulations} runs... {num_success_so_far} successful so far")

    # final stat
    num_success = sum(1 for res in results if res['status'] == 'Intercept')
    success_rate = num_success / num_simulations * 100 if num_simulations > 0 else 0
    
    # Calculate average steps
    successful_runs = [res for res in results if res['status'] == 'Intercept']
    failed_runs = [res for res in results if res['status'] != 'Intercept']
    
    avg_steps_success = np.mean([res['steps'] for res in successful_runs]) if successful_runs else 0
    avg_steps_failure = np.mean([res['steps'] for res in failed_runs]) if failed_runs else 0
    
    # Failure analysis to train futute models
    failure_types = {}
    for res in results:
        if res['status'] != 'Intercept':
            failure_types[res['status']] = failure_types.get(res['status'], 0) + 1
    
    print(f"\n--- RL Model Test Statistics ({num_simulations} runs) ---")
    print(f"Successful interceptions: {num_success}")
    print(f"Failed runs: {len(failed_runs)}")
    print(f"Success rate: {success_rate:.2f}%")
    print(f"Average steps to intercept: {avg_steps_success:.1f}")
    print(f"Average steps before failure: {avg_steps_failure:.1f}")
    
    if failure_types:
        print(f"\n--- Failure Analysis ---")
        for failure_type, count in failure_types.items():
            percentage = (count / num_simulations) * 100
            print(f"{failure_type}: {count} times ({percentage:.2f}%)")