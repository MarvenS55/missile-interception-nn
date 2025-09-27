import numpy as np

class PointMass:
    """A simple class that representss an object with position and Velocity."""
    def __init__(self, position, velocity, name="Object"):
        self.name = name
        # make surre position and velocity are Numpy arrays for vector operations
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        # We assume a mass of 1 for simplicity, so force = acceleration we can make this more complex later on
        self.mass = 1.0 

    def update(self, force, dt):
        """
        Updates the object's state using euler integrationn.
        F = ma -> a = F/m.
        """
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
