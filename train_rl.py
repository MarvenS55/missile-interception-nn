import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from missile_env.environment import MissileEnv

log_dir = "logs/"
model_dir = "models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

env = make_vec_env(MissileEnv, n_envs=8)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

model = PPO(
    'MlpPolicy', 
    env, 
    verbose=1, 
    tensorboard_log=log_dir,
    gamma=0.999,
    n_steps=4096,
    ent_coef=0.01
)

print("--- Starting Reinforcement Learning Training (V2) ---")
model.learn(total_timesteps=10000000)  # might be too much, after around 7 it gets useless, will test and udjust
print("--- Training Complete ---")

# Save the V2 model
model.save(os.path.join(model_dir, "ppo_missile_model_v2"))
env.save(os.path.join(model_dir, "vec_normalize_v2.pkl"))
print(f"Final model and normalization stats saved as v2.")