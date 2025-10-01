# missile_env/environment.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .dynamics import PointMass

class MissileEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        self.G = 9.81
        self.RHO = 1.225
        self.C_D = 0.5
        self.AREA = 0.0314
        self.time_step = 0.01
        self.max_steps = 10000
        self.current_step = 0
        self.max_thrust = 3000000.0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.interceptor = None
        self.target = None
        self.prev_distance = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        target_pos = [self.np_random.uniform(15000, 35000), self.np_random.uniform(5000, 25000), self.np_random.uniform(20000, 40000)]
        target_vel = [self.np_random.uniform(-600, -300), self.np_random.uniform(-150, 150), self.np_random.uniform(-100, 50)]
        
        self.target = PointMass(mass=500.0, position=target_pos, velocity=target_vel, name="Target")
        
        inter_pos_x = self.np_random.uniform(-1000, 1000)
        inter_pos_y = self.np_random.uniform(-1000, 1000)
        inter_pos_z = 100.0 + self.np_random.uniform(-50, 50)
        inter_vel_z = self.np_random.uniform(400, 600)
        self.interceptor = PointMass(mass=200.0, position=[inter_pos_x, inter_pos_y, inter_pos_z], velocity=[0.0, 0.0, inter_vel_z], name="Interceptor")
        
        observation = self._get_obs()
        info = self._get_info()
        
        self.prev_distance = np.linalg.norm(self.target.position - self.interceptor.position)
        
        return observation, info

    def step(self, action):
        interceptor_thrust = action * self.max_thrust
        
        interceptor_gravity, interceptor_drag = self._calculate_forces(self.interceptor)
        self.interceptor.update(interceptor_thrust, interceptor_gravity, interceptor_drag, self.time_step)

        target_gravity, target_drag = self._calculate_forces(self.target)
        self.target.update(np.zeros(3), target_gravity, target_drag, self.time_step)

        self.current_step += 1
        
        observation = self._get_obs()
        info = self._get_info()
        distance = info['distance']

        # --- REWARD: Based on distance reduction (other systems created a very lazy intercepter that just flopped)---
        delta_dist = distance - self.prev_distance  # negative iff distance reduced
        reward = -delta_dist - 0.1  # Positive if distance reduced, minus time penalty
        self.prev_distance = distance

        terminated = False
        if distance < 20.0:
            terminated = True
            reward += 1000
            info['status'] = 'Intercept'
        elif self.target.position[2] <= 0 or self.interceptor.position[2] <= 0:
            terminated = True
            reward -= 1000
            info['status'] = 'FAILED: Interceptor Impact' if self.interceptor.position[2] <= 0 else 'FAILED: Target Impact'
        
        truncated = self.current_step >= self.max_steps
        if truncated and not terminated:
            info['status'] = 'FAILED: TIMEOUT'
            reward -= 1000

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        relative_position = self.target.position - self.interceptor.position
        relative_velocity = self.target.velocity - self.interceptor.velocity
        return np.concatenate([relative_position, relative_velocity]).astype(np.float32)

    def _get_info(self):
        distance = np.linalg.norm(self.target.position - self.interceptor.position)
        return {'distance': distance, 'status': 'In-flight'}

    def _calculate_forces(self, obj):
        gravity_force = np.array([0.0, 0.0, -self.G * obj.mass])
        speed = np.linalg.norm(obj.velocity)
        drag_force = np.zeros(3)
        if speed > 0:
            drag_magnitude = 0.5 * self.RHO * speed**2 * self.C_D * self.AREA
            drag_force = -drag_magnitude * (obj.velocity / speed)
        return gravity_force, drag_force