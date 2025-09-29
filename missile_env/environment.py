# Updated environment.py with better initialization and interceptor ground impact check
import numpy as np
from .dynamics import PointMass

class MissileEnv:
    """
    Manages the simulation state, now including gravity and drag calculations.
    """
    def __init__(self):
        # --- Physical constants ---
        self.G = 9.81  # Acceleration due to gravity (m/s^2)
        self.RHO = 1.225  # Air density at sea level (kg/m^3)
        self.C_D = 0.5  # Drag coefficient (simplified)
        self.AREA = 0.0314 # Cross-sectional area of missile (m^2)

        self.interceptor = None
        self.target = None
        self.time_step = 0.05  # Using a smaller time step for more stability
        self.reset()

    def reset(self, target_pos=None, target_vel=None):
        """Resets the simulation with new mass properties and optional randomization."""
        if target_pos is None:
            # Ensure target is always hight above interceptor so it simulates it as approaching like real life 
            target_pos = [
                np.random.uniform(15000, 35000),  # Further horizontally
                np.random.uniform(5000, 25000),   # Wider yrange
                np.random.uniform(20000, 40000)   #  higher altitude
            ]
        
        if target_vel is None:
            # Ensure target has reasonable velocity (not diving too fast)
            target_vel = [
                np.random.uniform(-600, -300),    # Moderate xvelocity
                np.random.uniform(-150, 150),     #  Reasoonable yvelocity  
                np.random.uniform(-100, 50)       # Limitted descent rate
            ]
        
        # Target: Heavier, non-powered like irl
        self.target = PointMass(
            mass=500.0,  # in kg
            position=target_pos,
            velocity=target_vel,
            name="Target"
        )
        
        # Interceptor: Lighter, powered - starts with higher initial velocity
        self.interceptor = PointMass(
            mass=200.0,  # in kg
            position=[0.0, 0.0, 100.0],  # Start slightly above ground
            velocity=[0.0, 0.0, 500.0],  # higher initial velocity
            name="Interceptor"
        )
        
        return self.get_state()

    def get_state(self):
        """Returns the current state (relative position and velocity)."""
        relative_position = self.target.position - self.interceptor.position
        relative_velocity = self.target.velocity - self.interceptor.velocity
        return np.concatenate([relative_position, relative_velocity])

    def _calculate_forces(self, obj):
        """Helper function to calculate gravity and drag for a PointMass object."""
        # Gravity force (acts downwards on Z axis)
        gravity_force = np.array([0.0, 0.0, -self.G * obj.mass])

        # Drag force (opposes velocity vector)
        speed = np.linalg.norm(obj.velocity)
        if speed > 0:
            drag_magnitude = 0.5 * self.RHO * speed**2 * self.C_D * self.AREA
            drag_force = -drag_magnitude * (obj.velocity / speed)
        else:
            drag_force = np.array([0.0, 0.0, 0.0])

        return gravity_force, drag_force

    def step(self, interceptor_thrust):
        """Advances the simulation by one time step."""
        # Calculate forces for the interceptor
        interceptor_gravity, interceptor_drag = self._calculate_forces(self.interceptor)
        self.interceptor.update(interceptor_thrust, interceptor_gravity, interceptor_drag, self.time_step)

        # Calculate forces for the target (no thrust)
        target_gravity, target_drag = self._calculate_forces(self.target)
        self.target.update(np.array([0,0,0]), target_gravity, target_drag, self.time_step)

        new_state = self.get_state()
        
        distance = np.linalg.norm(self.target.position - self.interceptor.position)
        done = False
        info = {'status': 'In-flight', 'distance': distance}

        if distance < 20.0:
            done = True
            info['status'] = 'Intercept'
        
        # Failure condition: target hits the ground
        if self.target.position[2] <= 0:
            done = True
            info['status'] = 'FAILED: Target Impact'
            
        # New failure condition: interceptor hits the ground
        if self.interceptor.position[2] <= 0:
            done = True
            info['status'] = 'FAILED: Interceptor Impact'

        return new_state, done, info
