"""
Vehicle simulation module
Handles physics, sensors, collision detection, and state
"""

import math
from src.utils import raycast_to_track, normalize_distance, Vector2


class Car:
    """Represents a racing car with physics and sensors"""

    # Physics constants
    MAX_SPEED = 300  # pixels per frame at max throttle
    ACCELERATION = 10  # pixels per frame^2
    FRICTION = 0.98  # velocity decay when no throttle
    MAX_STEERING_RATE = 0.1  # radians per frame
    CAR_SIZE = 10  # collision radius in pixels

    def __init__(self, car_id, track, start_x, start_y, start_heading=0):
        """
        Initialize a car

        Args:
            car_id: Unique identifier for this car
            track: Track object for collision checking
            start_x, start_y: Starting position
            start_heading: Starting heading in radians
        """
        self.car_id = car_id
        self.track = track

        # State
        self.x = start_x
        self.y = start_y
        self.heading = start_heading  # radians
        self.velocity = 0  # pixels per frame
        self.alive = True

        # Sensor data (5 rays)
        self.sensor_values = [0] * 5  # normalized distances (0..1)
        self.sensor_angles = [
            -math.pi / 4,  # -45°
            -math.pi / 8,  # -22.5°
            0,  # straight
            math.pi / 8,  # +22.5°
            math.pi / 4,  # +45°
        ]

        # Fitness tracking
        self.total_distance = 0
        self.lifetime_steps = 0
        self.total_speed_accumulated = 0

        # Prime sensors at spawn so the first network decision uses real distances.
        self._update_sensors()

    def update(self, steering_input, throttle_input, dt=1.0):
        """
        Update car physics

        Args:
            steering_input: -1..1, steering command (-1 = left, +1 = right)
            throttle_input: -1..1, throttle command (-1 = brake, 0 = neutral, +1 = gas)
            dt: delta time (default 1 frame)
        """
        if not self.alive:
            return

        # Apply throttle
        if throttle_input > 0:
            # Accelerate
            self.velocity += self.ACCELERATION * throttle_input * dt
            self.velocity = min(self.velocity, self.MAX_SPEED)
        elif throttle_input < 0:
            # Brake
            self.velocity += self.ACCELERATION * throttle_input * dt
            self.velocity = max(self.velocity, 0)
        else:
            # Coast with friction
            self.velocity *= self.FRICTION

        # Apply steering
        steering_change = steering_input * self.MAX_STEERING_RATE * dt
        self.heading += steering_change

        # Update position
        dx = math.cos(self.heading) * self.velocity
        dy = math.sin(self.heading) * self.velocity

        new_x = self.x + dx
        new_y = self.y + dy

        # Check collision with track boundary
        if not self.track.is_on_track(new_x, new_y):
            # Car went off track - DEAD
            self.alive = False
            return

        # Update position
        self.x = new_x
        self.y = new_y

        # Update totals for fitness
        self.total_distance += self.velocity
        self.lifetime_steps += 1
        self.total_speed_accumulated += self.velocity

        # Update sensors
        self._update_sensors()

    def _update_sensors(self):
        """Update sensor values by raycasting to track boundaries"""
        sensor_range = self.track.get_sensor_range()
        track_mask = self.track.get_mask()

        for i, angle_offset in enumerate(self.sensor_angles):
            # Ray direction = heading + offset
            ray_direction = self.heading + angle_offset

            # Cast ray
            distance = raycast_to_track(
                (self.x, self.y),
                ray_direction,
                track_mask,
                sensor_range,
                step_size=2,
            )

            # Normalize to 0..1 range
            self.sensor_values[i] = normalize_distance(distance, sensor_range)

    def get_sensor_readings(self):
        """Return normalized sensor values (0..1 for each of 5 sensors)"""
        return self.sensor_values.copy()

    def get_position(self):
        """Return (x, y) position"""
        return (self.x, self.y)

    def get_heading(self):
        """Return heading in radians"""
        return self.heading

    def get_velocity(self):
        """Return current velocity"""
        return self.velocity

    def is_alive(self):
        """Return whether car is still on track"""
        return self.alive

    def get_average_speed(self):
        """Return average speed over lifetime"""
        if self.lifetime_steps == 0:
            return 0
        return self.total_speed_accumulated / self.lifetime_steps

    def get_lifetime_steps(self):
        """Return steps car has been alive"""
        return self.lifetime_steps

    def get_sensor_angles(self):
        """Return the angles of the 5 sensors relative to heading (for debugging)"""
        return self.sensor_angles

    def reset(self, x, y, heading=0):
        """Reset car to initial state at given position"""
        self.x = x
        self.y = y
        self.heading = heading
        self.velocity = 0
        self.alive = True
        self.total_distance = 0
        self.lifetime_steps = 0
        self.total_speed_accumulated = 0
        self._update_sensors()
