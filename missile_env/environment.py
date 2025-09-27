import numpy as np
from .dynamics import PointMass

class MissileEnv:
    """Manages the overall simulation state.."""
    def __init__(self):
        self.interceptor = None
        self.target = None
        self.time_step = 0.05
        self.reset()

    def reset(self):
        """Resets the simulation to a more realistic scenario."""
        # Target starts high and far, moving horizontally as if its coming from far away
        self.target = PointMass(position=[20000.0, 10000.0, 0.0], velocity=[-400.0, 0.0, 0.0], name="Target")
        
        # interceptor starts on the ground (0,0,0) with an initial upward launch velocity.
        self.interceptor = PointMass(position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 50.0], name="Interceptor")
        
        return self._get_state()

    def _get_state(self):
        """Returns the current state of the environment ."""
        relative_position = self.target.position - self.interceptor.position
        relative_velocity = self.target.velocity - self.interceptor.velocity
        return np.concatenate([relative_position, relative_velocity])

    def step(self, action_force):
        """Advances the simulation by one time step."""
        self.interceptor.update(action_force, self.time_step)
        self.target.update(np.array([0.0, 0.0, 0.0]), self.time_step)

        new_state = self._get_state()
        distance = np.linalg.norm(self.target.position - self.interceptor.position)
        
        done = False
        info = {'status': 'In-flight'}

        if distance < 20.0: # Increased intercept radius will lower later
            done = True
            info['status'] = 'Intercept'
        
        return new_state, done, info

