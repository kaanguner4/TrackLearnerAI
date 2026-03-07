"""
Utility functions for math, raycasting, and progress tracking
"""

import math
import numpy as np


class Vector2:
    """2D vector utility class"""

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)

    def length(self):
        return math.sqrt(self.x**2 + self.y**2)

    def normalized(self):
        length = self.length()
        if length == 0:
            return Vector2(0, 0)
        return Vector2(self.x / length, self.y / length)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def to_tuple(self):
        return (self.x, self.y)


def normalize_angle(angle):
    """Normalize angle to 0..2π range"""
    while angle < 0:
        angle += 2 * math.pi
    while angle >= 2 * math.pi:
        angle -= 2 * math.pi
    return angle


def angle_difference(angle1, angle2):
    """Calculate shortest angle difference in radians (-π to π)"""
    diff = angle2 - angle1
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff


def raycast_to_track(
    start_pos, direction, track_mask, max_distance=500, step_size=1
):
    """
    Cast a ray from start_pos in direction until hitting track boundary

    Args:
        start_pos: (x, y) tuple, starting position
        direction: direction angle in radians
        track_mask: pygame.mask object representing track obstacles
        max_distance: maximum ray distance
        step_size: pixel step size for raycasting

    Returns:
        distance to hit (0..max_distance), or max_distance if no hit
    """
    x, y = start_pos
    dx = math.cos(direction)
    dy = math.sin(direction)

    distance = 0

    while distance < max_distance:
        # Current position along ray
        curr_x = int(x + dx * distance)
        curr_y = int(y + dy * distance)

        # Check if we hit a boundary (mask pixel is True = black boundary)
        try:
            if curr_x < 0 or curr_y < 0:
                return distance

            if track_mask.get_at((curr_x, curr_y)):
                # Hit a boundary (black pixel)
                return distance
        except (IndexError, ValueError):
            # Out of bounds = hit boundary
            return distance

        distance += step_size

    return max_distance


def normalize_distance(distance, max_distance):
    """Normalize distance to 0..1 range (1 = max_distance reached)"""
    return min(1.0, distance / max_distance)


class ProgressTracker:
    """
    Tracks car progress along track using checkpoint system
    Prevents "doughnut driving" (circles in place) by using monotonic progress
    """

    def __init__(self, track_width, track_height, num_checkpoints=50):
        """
        Initialize progress tracker

        Args:
            track_width, track_height: Track dimensions
            num_checkpoints: Number of segments to divide track into
        """
        self.num_checkpoints = num_checkpoints
        self.track_width = track_width
        self.track_height = track_height

        # Generate checkpoint positions along an oval path
        # Simplified: checkpoints arranged in a circle
        self.checkpoints = self._generate_checkpoints()

        # For each car, track furthest checkpoint reached
        self.furthest_checkpoint = {}

    def _generate_checkpoints(self):
        """Generate checkpoint positions around track oval"""
        checkpoints = []
        center_x = self.track_width / 2
        center_y = self.track_height / 2

        # Ellipse parameters (should match track dimensions roughly)
        radius_x = self.track_width * 0.3
        radius_y = self.track_height * 0.25

        for i in range(self.num_checkpoints):
            angle = (2 * math.pi * i) / self.num_checkpoints
            x = center_x + radius_x * math.cos(angle)
            y = center_y + radius_y * math.sin(angle)
            checkpoints.append((x, y))

        return checkpoints

    def calculate_progress(self, car_id, car_pos):
        """
        Calculate monotonic progress score for a car

        Args:
            car_id: Unique car identifier
            car_pos: (x, y) current position

        Returns:
            progress value (0..num_checkpoints, monotonically increasing)
        """
        if car_id not in self.furthest_checkpoint:
            self.furthest_checkpoint[car_id] = 0

        # Find closest checkpoint
        car_x, car_y = car_pos
        min_distance = float("inf")
        closest_checkpoint = 0

        for i, (cp_x, cp_y) in enumerate(self.checkpoints):
            distance = math.sqrt((car_x - cp_x) ** 2 + (car_y - cp_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_checkpoint = i

        # Check if we've crossed into a new checkpoint
        furthest = self.furthest_checkpoint[car_id]

        # Handle wrap-around (checkpoint 0 after checkpoint n-1)
        if closest_checkpoint > furthest:
            self.furthest_checkpoint[car_id] = closest_checkpoint
        elif closest_checkpoint < furthest and furthest > self.num_checkpoints - 5:
            # Likely wrapped around to checkpoint 0
            self.furthest_checkpoint[car_id] = closest_checkpoint + self.num_checkpoints

        # Add linear interpolation between checkpoints
        next_checkpoint = (self.furthest_checkpoint[car_id] + 1) % self.num_checkpoints
        curr_cp = self.checkpoints[self.furthest_checkpoint[car_id] % self.num_checkpoints]
        next_cp = self.checkpoints[next_checkpoint]

        # Distance to next checkpoint
        curr_x, curr_y = car_pos
        curr_cp_x, curr_cp_y = curr_cp
        next_cp_x, next_cp_y = next_cp

        dist_to_curr = math.sqrt((curr_x - curr_cp_x) ** 2 + (curr_y - curr_cp_y) ** 2)
        dist_curr_to_next = math.sqrt(
            (next_cp_x - curr_cp_x) ** 2 + (next_cp_y - curr_cp_y) ** 2
        )

        if dist_curr_to_next < 0.1:
            interpolation = 0
        else:
            interpolation = min(1.0, dist_to_curr / dist_curr_to_next)

        return self.furthest_checkpoint[car_id] + interpolation

    def get_progress(self, car_id):
        """Get current progress (0..num_checkpoints) for a car"""
        if car_id not in self.furthest_checkpoint:
            return 0
        return self.furthest_checkpoint[car_id]

    def reset_car(self, car_id):
        """Reset progress for a car"""
        if car_id in self.furthest_checkpoint:
            del self.furthest_checkpoint[car_id]


def calculate_fitness(progress, avg_speed, is_dead, collision_penalty=50):
    """
    Calculate fitness score for a car

    Formula: fitness = progress + (avg_speed × 0.5) - collision_penalty (if dead)

    Args:
        progress: how far on track (0..50 typically)
        avg_speed: average speed maintained
        is_dead: whether car crashed
        collision_penalty: penalty for collision

    Returns:
        fitness score
    """
    fitness = progress + (avg_speed * 0.5)

    if is_dead:
        fitness -= collision_penalty

    return max(0, fitness)
