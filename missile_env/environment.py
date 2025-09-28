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
        self.C_D = 0.5  # Drag coefficient (simplified) will add complecity later
        self.AREA = 0.0314 # Cross sectional area of missile (m^2)

        self.interceptor = None
        self.target = None
        self.time_step = 0.05  # Using a smaller time step for more stability
        self.reset()

    def reset(self):
        """Resets the simulation with new mass properties."""
        # Target: Heavier, non-powered
        self.target = PointMass(
            mass=500.0,  # in kg
            position=[20000.0, 10000.0, 15000.0],
            velocity=[-500.0, 0.0, -150.0], # Starts with a downward trajectory to simulate it approaching
            name="Target"
        )
        # Interceptor: Lighter, powered
        self.interceptor = PointMass(
            mass=200.0, # in kg
            position=[0.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 300.0], # Initial vertical launch velocity
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

        # Drag force (opposes velocity vector) just in the opposite direction in all cases to account for missile to turn around during chase
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
        info = {'status': 'In-flight'}

        if distance < 20.0:
            done = True
            info['status'] = 'Intercept'
        
        # New failure condition: target hits the ground (very bad)
        if self.target.position[2] <= 0:
            done = True
            info['status'] = 'FAILED: Target Impact'

        return new_state, done, info

