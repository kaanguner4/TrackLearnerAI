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
    Tracks car progress using track-defined checkpoint guides.
    Blue checkpoints must be collected in order, clockwise from red start/finish.
    """

    def __init__(self, track_or_width, track_height=None, num_checkpoints=50):
        """
        Initialize progress tracker

        Args:
            track_or_width: Track object (preferred) or legacy width integer
            track_height: Legacy mode height
            num_checkpoints: Legacy mode checkpoint count
        """
        self.track = None
        self.car_states = {}

        # Preferred mode: parse progress from track's blue checkpoint lines.
        if hasattr(track_or_width, "get_crossed_checkpoints"):
            self.track = track_or_width
            self.num_checkpoints = max(1, self.track.get_checkpoint_count())
            self.track_width = self.track.width
            self.track_height = self.track.height
            self.checkpoints = self.track.get_checkpoint_centers()
            self.furthest_checkpoint = {}
            return

        # Legacy fallback mode.
        self.num_checkpoints = num_checkpoints
        self.track_width = track_or_width
        self.track_height = track_height
        self.checkpoints = self._generate_legacy_checkpoints()
        self.furthest_checkpoint = {}

    def _generate_legacy_checkpoints(self):
        """Generate fallback checkpoints around an ellipse."""
        checkpoints = []
        center_x = self.track_width / 2
        center_y = self.track_height / 2

        radius_x = self.track_width * 0.3
        radius_y = self.track_height * 0.25

        for i in range(self.num_checkpoints):
            angle = (2 * math.pi * i) / self.num_checkpoints
            x = center_x + radius_x * math.cos(angle)
            y = center_y + radius_y * math.sin(angle)
            checkpoints.append((x, y))

        return checkpoints

    def _init_car_state(self, car_id, car_pos):
        """Initialize per-car state for checkpoint collection mode."""
        if car_id in self.car_states:
            return self.car_states[car_id]

        next_checkpoint = self._get_initial_next_checkpoint(car_pos)
        self.car_states[car_id] = {
            "next_checkpoint": next_checkpoint,
            "collected": 0,
            "laps": 0,
            "wrong_way_hits": 0,
            "last_on_finish": False,
        }
        return self.car_states[car_id]

    def _get_initial_next_checkpoint(self, car_pos):
        """Choose the first checkpoint that lies ahead in clockwise direction."""
        if self.track is None or len(self.checkpoints) == 0:
            return 0

        heading = self.track.get_clockwise_vector()
        nearest_idx = min(
            range(len(self.checkpoints)),
            key=lambda idx: math.hypot(
                self.checkpoints[idx][0] - car_pos[0],
                self.checkpoints[idx][1] - car_pos[1],
            ),
        )

        nearest_checkpoint = self.checkpoints[nearest_idx]
        dx = nearest_checkpoint[0] - car_pos[0]
        dy = nearest_checkpoint[1] - car_pos[1]
        projection = dx * heading[0] + dy * heading[1]

        if projection >= 0:
            return nearest_idx

        return (nearest_idx + 1) % len(self.checkpoints)

    def calculate_progress(self, car_id, car_pos, previous_pos=None):
        """
        Calculate monotonic progress score for a car.

        Args:
            car_id: Unique car identifier
            car_pos: (x, y) current position
            previous_pos: (x, y) previous position for crossing detection

        Returns:
            progress value (monotonically increasing)
        """
        if self.track is None:
            return self._calculate_legacy_progress(car_id, car_pos)

        if previous_pos is None:
            previous_pos = car_pos

        state = self._init_car_state(car_id, car_pos)

        crossed_checkpoints = self.track.get_crossed_checkpoints(previous_pos, car_pos)
        for checkpoint_idx in crossed_checkpoints:
            expected = state["next_checkpoint"]

            if checkpoint_idx == expected:
                state["collected"] += 1
                state["next_checkpoint"] = (expected + 1) % self.num_checkpoints
            elif checkpoint_idx == (expected - 1) % self.num_checkpoints:
                # Ignore tiny bounce-backs on the same line.
                continue
            else:
                # Out-of-order crossing = likely wrong-way movement.
                state["wrong_way_hits"] += 1

        crossed_finish = self.track.segment_crosses_start_finish(
            previous_pos,
            car_pos,
            require_clockwise=True,
        )
        on_finish = self.track.is_on_start_finish(car_pos[0], car_pos[1])
        if crossed_finish and not state["last_on_finish"]:
            target_collected = (state["laps"] + 1) * self.num_checkpoints
            if state["collected"] >= target_collected:
                state["laps"] += 1
        state["last_on_finish"] = on_finish

        progress = float(state["collected"])

        # Add small interpolation towards the next checkpoint for smoother fitness.
        next_center = self.track.get_checkpoint_center(state["next_checkpoint"])
        if next_center is not None:
            distance = math.hypot(
                car_pos[0] - next_center[0],
                car_pos[1] - next_center[1],
            )
            max_distance = max(1.0, float(max(self.track_width, self.track_height)))
            progress += max(0.0, 1.0 - (distance / max_distance))

        progress -= state["wrong_way_hits"] * 0.25
        return max(0.0, progress)

    def _calculate_legacy_progress(self, car_id, car_pos):
        """Legacy ellipse-based progress calculation."""
        if car_id not in self.furthest_checkpoint:
            self.furthest_checkpoint[car_id] = 0

        car_x, car_y = car_pos
        min_distance = float("inf")
        closest_checkpoint = 0

        for i, (cp_x, cp_y) in enumerate(self.checkpoints):
            distance = math.sqrt((car_x - cp_x) ** 2 + (car_y - cp_y) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_checkpoint = i

        furthest = self.furthest_checkpoint[car_id]

        if closest_checkpoint > furthest:
            self.furthest_checkpoint[car_id] = closest_checkpoint
        elif closest_checkpoint < furthest and furthest > self.num_checkpoints - 5:
            self.furthest_checkpoint[car_id] = closest_checkpoint + self.num_checkpoints

        next_checkpoint = (self.furthest_checkpoint[car_id] + 1) % self.num_checkpoints
        curr_cp = self.checkpoints[self.furthest_checkpoint[car_id] % self.num_checkpoints]
        next_cp = self.checkpoints[next_checkpoint]

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
        if self.track is not None:
            state = self.car_states.get(car_id)
            if state is None:
                return 0
            return state["collected"]

        if car_id not in self.furthest_checkpoint:
            return 0
        return self.furthest_checkpoint[car_id]

    def reset_car(self, car_id):
        """Reset progress for a car"""
        if self.track is not None and car_id in self.car_states:
            del self.car_states[car_id]

        if car_id in self.furthest_checkpoint:
            del self.furthest_checkpoint[car_id]


def calculate_fitness(progress, avg_speed, is_dead, collision_penalty=20):
    """
    Calculate fitness score for a car

    Formula: fitness = (progress × 10) + (avg_speed × 0.2) - collision_penalty (if dead)

    Args:
        progress: checkpoint progress (blue lines collected in order)
        avg_speed: average speed maintained
        is_dead: whether car crashed
        collision_penalty: penalty for collision

    Returns:
        fitness score
    """
    fitness = (progress * 10.0) + (avg_speed * 0.2)

    if is_dead:
        fitness -= collision_penalty

    return max(0, fitness)
