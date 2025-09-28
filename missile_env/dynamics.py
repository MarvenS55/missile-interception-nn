import numpy as np

class PointMass:
    """
    Represents an object with mass, position, and velocity.
    Now includes properties for calculating gravity and drag.
    """
    def __init__(self, mass, position, velocity, name="Object"):
        self.name = name
        self.mass = mass  # Mass in kg
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

    def update(self, thrust_force, gravity_force, drag_force, dt):
        """
        Updates the object's state using Euler integratioon.
        calculates total force and updates acceleratioon.
        F_total = F_thrust + F_gravity + F_drag
        a = F_total / m
        """
        # --- Total force calculation ---
        total_force = thrust_force + gravity_force + drag_force
        
        # --- Acceleration now depends on mass ---
        acceleration = total_force / self.mass
        
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

